"""Encode the corpus with a dense bi-encoder (E5-base-v2), in resumable shards.

Each document becomes one 768-number vector ("embedding") that captures its
meaning, so documents can later be matched to queries that use different words.

  text fed to the model:  "passage: " + title + " " + body    (E5 needs the prefix)
  truncation:             max_length tokens (512) incl. prefix and title
  vectors:                L2-normalized, so inner product = cosine similarity

The corpus is split into shards of shard_size documents. Each finished shard is
saved to Drive immediately (vectors .npy + doc ids .txt + timing .json), and
shards that already exist are skipped. If Colab disconnects, rerunning the same
command continues where it stopped.

Output: <data_root>/<dataset>/dense/<model>/shard_0000.npy, shard_0000.ids.txt, ...
        <data_root>/<dataset>/dense/<model>.manifest.json

  python -m src.dense_encode --dataset trec_covid      (GPU runtime)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from src.config import add_config_args, config_from_args, data_dir
from src.io_utils import write_manifest

Encoder = Callable[[list[str]], np.ndarray]  # texts -> (n, dim) float32, rows L2-normalized


# ---------------------------------------------------------------- paths

def model_slug(model_name: str) -> str:
    """intfloat/e5-base-v2 -> e5-base-v2 (folder name)."""
    return model_name.rstrip("/").split("/")[-1]


def shards_dir(cfg: dict) -> Path:
    return data_dir(cfg, "dense", model_slug(cfg["dense"]["model"]))


def encode_manifest_path(cfg: dict) -> Path:
    return data_dir(cfg, "dense", f"{model_slug(cfg['dense']['model'])}.manifest.json")


def shard_paths(out_dir: Path, i: int) -> tuple[Path, Path, Path]:
    stem = out_dir / f"shard_{i:04d}"
    return stem.with_suffix(".npy"), stem.with_suffix(".ids.txt"), stem.with_suffix(".json")


# ---------------------------------------------------------------- text

def passage_text(record: dict, prefix: str) -> str:
    return f"{prefix}{record['title']} {record['body']}".strip()


def batched(items: Iterable, size: int) -> Iterator[list]:
    it = iter(items)
    while batch := list(itertools.islice(it, size)):
        yield batch


# ---------------------------------------------------------------- encoding

def encode_shards(records: Iterable[dict], encode: Encoder, out_dir: Path,
                  shard_size: int, prefix: str) -> dict:
    """Encode records shard by shard, skipping shards already on disk.

    A shard counts as finished only when its timing .json exists; that file is
    written last, so a shard interrupted mid-save is redone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = n_skipped = 0
    for i, shard in enumerate(batched(records, shard_size)):
        vec_path, ids_path, info_path = shard_paths(out_dir, i)
        if info_path.exists():
            n_skipped += 1
            continue
        start = time.perf_counter()
        vectors = np.asarray(encode([passage_text(r, prefix) for r in shard]), dtype=np.float32)
        if vectors.shape[0] != len(shard):
            raise RuntimeError(f"Shard {i}: got {vectors.shape[0]} vectors for {len(shard)} documents")
        seconds = time.perf_counter() - start

        _save_atomic(vec_path, lambda f: np.save(f, vectors))
        _save_atomic(ids_path, lambda f: f.write(("\n".join(r["id"] for r in shard) + "\n").encode("utf-8")))
        _save_atomic(info_path, lambda f: f.write(
            json.dumps({"n_docs": len(shard), "seconds": round(seconds, 2)}).encode("utf-8")))
        n_new += 1
        print(f"shard {i}: {len(shard)} docs in {seconds:.1f}s", flush=True)
    return {"shards_encoded_now": n_new, "shards_already_done": n_skipped}


def _save_atomic(path: Path, save) -> None:
    """Write via a temp file + rename, so a half-written file never has the final name."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        save(f)
    os.replace(tmp, path)


def load_shards(out_dir: Path) -> tuple[list[str], np.ndarray]:
    """All doc ids and vectors in shard order, as one (n_docs, dim) float32 matrix."""
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    i = 0
    while True:
        vec_path, ids_path, info_path = shard_paths(out_dir, i)
        if not info_path.exists():
            break
        shard_ids = ids_path.read_text(encoding="utf-8").split()
        vectors = np.load(vec_path)
        if len(shard_ids) != vectors.shape[0]:
            raise RuntimeError(f"Shard {i}: {len(shard_ids)} ids but {vectors.shape[0]} vectors")
        ids.extend(shard_ids)
        blocks.append(vectors)
        i += 1
    if not blocks:
        raise FileNotFoundError(f"No finished shards in {out_dir}; run src.dense_encode first")
    return ids, np.concatenate(blocks).astype(np.float32, copy=False)


def shard_stats(out_dir: Path) -> dict:
    infos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("shard_*.json"))]
    size = sum(p.stat().st_size for p in out_dir.glob("shard_*.npy"))
    return {
        "n_shards": len(infos),
        "n_docs": sum(x["n_docs"] for x in infos),
        "encode_seconds_total": round(sum(x["seconds"] for x in infos), 1),
        "embeddings_size_mb": round(size / 2**20, 1),
    }


# ---------------------------------------------------------------- model

def load_model(dense_cfg: dict):
    """SentenceTransformer for the configured model, on GPU in fp16 when available."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(dense_cfg["model"], revision=dense_cfg.get("revision"), device=device)
    model.max_seq_length = dense_cfg["max_length"]
    if dense_cfg["fp16"] and device == "cuda":
        model.half()
    return model, device


def make_encoder(model, dense_cfg: dict) -> Encoder:
    def encode(texts: list[str]) -> np.ndarray:
        # prompt="" stops sentence-transformers adding a built-in prompt on top of
        # the prefix we already added ourselves.
        return model.encode(
            texts, batch_size=dense_cfg["batch_size"], prompt="",
            normalize_embeddings=dense_cfg["normalize"], convert_to_numpy=True, show_progress_bar=False,
        )
    return encode


def resolved_revision(model_name: str, revision: str | None) -> str | None:
    """The exact model commit used, for the manifest (proposal §5.1: freeze model revision)."""
    try:
        from huggingface_hub import model_info
        return model_info(model_name, revision=revision).sha
    except Exception:  # offline or API change: record nothing rather than fail the run
        return None


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")
    dense = cfg["dense"]

    from src.prepare_corpus import corpus_path, read_corpus

    model, device = load_model(dense)
    print(f"Model {dense['model']} on {device}; max_seq_length={model.max_seq_length}")
    out_dir = shards_dir(cfg)
    progress = encode_shards(
        read_corpus(corpus_path(cfg)), make_encoder(model, dense), out_dir,
        dense["shard_size"], dense["passage_prefix"],
    )
    stats = shard_stats(out_dir)
    write_manifest(
        encode_manifest_path(cfg), cfg, **progress, **stats, device=device,
        gpu=_gpu_name(), model_revision=resolved_revision(dense["model"], dense.get("revision")),
        embedding_dim=model.get_sentence_embedding_dimension(),
    )
    print(json.dumps({**progress, **stats}, indent=2))


def _gpu_name() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


if __name__ == "__main__":
    main()
