"""Dense retrieval: find the documents whose embeddings are closest to the query's.

  1. Load the document vectors saved by src.dense_encode into a FAISS
     IndexFlatIP: exact (brute-force) inner-product search. Vectors are
     L2-normalized, so inner product = cosine similarity.
  2. Encode each query as "query: " + text with the same model.
  3. Return the top `depth` documents per query as a TREC run.

Latency per query = query encoding + search, measured one query at a time after
an untimed warm-up, the same way as BM25.

  python -m src.dense_search --dataset trec_covid
  python -m src.dense_search --dataset trec_covid --queries queries/trec_covid_text.tsv

Output: runs/<dataset>/dense_<query set>.run (+ .manifest.json)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.config import add_config_args, config_from_args, repo_dir
from src.dense_encode import Encoder, encode_manifest_path, load_model, load_shards, make_encoder, shards_dir
from src.io_utils import Run, manifest_path, read_queries, write_manifest, write_run
from src.load_topics import default_queries_path, query_set_name


def build_index(vectors: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def search_one(index, doc_ids: list[str], query_vector: np.ndarray, depth: int) -> dict[str, float]:
    scores, positions = index.search(np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32), depth)
    return {doc_ids[p]: float(s) for s, p in zip(scores[0], positions[0]) if p >= 0}


def search_queries(index, doc_ids: list[str], queries: dict[str, str], encode: Encoder,
                   prefix: str, depth: int) -> tuple[Run, list[float]]:
    if queries:  # warm-up: first GPU call and FAISS setup are not representative
        search_one(index, doc_ids, encode([prefix + next(iter(queries.values()))])[0], depth)
    run: Run = {}
    latencies_ms: list[float] = []
    for qid, text in queries.items():
        start = time.perf_counter()
        run[qid] = search_one(index, doc_ids, encode([prefix + text])[0], depth)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return run, latencies_ms


def run_search(cfg: dict, queries_path: Path | None = None) -> Path:
    from src.bm25 import latency_summary

    dense = cfg["dense"]
    name = cfg["dataset"]["name"]
    queries_path = queries_path or default_queries_path(cfg)
    queries = read_queries(queries_path)

    start = time.perf_counter()
    doc_ids, vectors = load_shards(shards_dir(cfg))
    index = build_index(vectors)
    index_seconds = time.perf_counter() - start

    model, device = load_model(dense)
    run, latencies = search_queries(
        index, doc_ids, queries, make_encoder(model, dense), dense["query_prefix"], dense["depth"],
    )

    run_tag = f"dense_{query_set_name(queries_path, name)}"
    out = write_run(run, repo_dir(cfg, "runs_dir") / name / f"{run_tag}.run", tag=run_tag, depth=dense["depth"])

    encode_info = {}
    if encode_manifest_path(cfg).exists():
        saved = json.loads(encode_manifest_path(cfg).read_text(encoding="utf-8"))
        encode_info = {k: saved.get(k) for k in (
            "encode_seconds_total", "embeddings_size_mb", "n_docs", "gpu", "model_revision", "embedding_dim")}
    write_manifest(
        manifest_path(out), cfg, system="dense", queries_file=str(queries_path), device=device,
        latency=latency_summary(latencies),
        index={"faiss_index": "IndexFlatIP", "n_vectors": index.ntotal,
               "load_and_build_seconds": round(index_seconds, 1),
               "index_memory_mb": round(vectors.nbytes / 2**20, 1)},
        encoding=encode_info,
    )
    print(f"{len(run)} queries on {device}, latency {latency_summary(latencies)}")
    print(f"Wrote {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--queries", type=Path, help="query TSV (default: the dataset's original queries)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")
    run_search(cfg, args.queries)


if __name__ == "__main__":
    main()
