"""Dense pipeline with a fake encoder (no model download): sharding, resuming,
loading and exact FAISS search. The real E5 model is exercised in Colab."""
import zlib

import numpy as np
import pytest

from src.dense_encode import (
    encode_shards, load_shards, model_slug, passage_text, shard_paths, shard_stats,
)

faiss = pytest.importorskip("faiss")
from src.dense_search import build_index, search_one, search_queries  # noqa: E402

DIM = 8


def fake_encode(texts):
    """Deterministic unit vector per text (same text -> same vector)."""
    out = []
    for text in texts:
        v = np.random.default_rng(zlib.crc32(text.encode())).normal(size=DIM)
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


RECORDS = [{"id": f"d{i}", "title": f"Title {i}", "body": f"body {i}"} for i in range(7)]


def test_model_slug_and_passage_text():
    assert model_slug("intfloat/e5-base-v2") == "e5-base-v2"
    assert passage_text({"title": "T", "body": "B"}, "passage: ") == "passage: T B"
    assert passage_text({"title": "T", "body": ""}, "passage: ") == "passage: T"


def test_shards_written_and_loaded_in_order(tmp_path):
    progress = encode_shards(RECORDS, fake_encode, tmp_path, shard_size=3, prefix="passage: ")
    assert progress == {"shards_encoded_now": 3, "shards_already_done": 0}  # 3 + 3 + 1 docs

    ids, vectors = load_shards(tmp_path)
    assert ids == [r["id"] for r in RECORDS]
    assert vectors.shape == (7, DIM) and vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[0], fake_encode([passage_text(RECORDS[0], "passage: ")])[0])
    assert shard_stats(tmp_path)["n_docs"] == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_skips_finished_shards(tmp_path):
    encode_shards(RECORDS, fake_encode, tmp_path, shard_size=3, prefix="passage: ")
    # Simulate a disconnect during shard 1: its timing file (written last) is missing.
    shard_paths(tmp_path, 1)[2].unlink()

    calls = []
    def counting_encode(texts):
        calls.append(len(texts))
        return fake_encode(texts)

    progress = encode_shards(RECORDS, counting_encode, tmp_path, shard_size=3, prefix="passage: ")
    assert progress == {"shards_encoded_now": 1, "shards_already_done": 2}
    assert calls == [3]  # only shard 1 was re-encoded
    assert load_shards(tmp_path)[0] == [r["id"] for r in RECORDS]


def test_load_shards_needs_at_least_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_shards(tmp_path)


def test_wrong_vector_count_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="vectors"):
        encode_shards(RECORDS, lambda texts: fake_encode(texts[:-1]), tmp_path, shard_size=3, prefix="")


def test_exact_search_finds_identical_vector(tmp_path):
    encode_shards(RECORDS, fake_encode, tmp_path, shard_size=3, prefix="passage: ")
    ids, vectors = load_shards(tmp_path)
    index = build_index(vectors)

    hits = search_one(index, ids, vectors[4], depth=3)
    assert list(hits)[0] == "d4"                      # a document is its own nearest neighbour
    assert hits["d4"] == pytest.approx(1.0, abs=1e-5)  # cosine of a unit vector with itself
    assert len(hits) == 3


def test_search_queries_uses_query_prefix(tmp_path):
    encode_shards(RECORDS, fake_encode, tmp_path, shard_size=3, prefix="passage: ")
    ids, vectors = load_shards(tmp_path)
    seen = []
    def recording_encode(texts):
        seen.extend(texts)
        return fake_encode(texts)

    run, latencies = search_queries(build_index(vectors), ids, {"1": "hello"}, recording_encode, "query: ", depth=2)
    assert seen[-1] == "query: hello"
    assert len(run["1"]) == 2 and len(latencies) == 1
