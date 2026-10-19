"""Reciprocal rank fusion (RRF): combine several runs into one hybrid run.

For every document d, across the input runs:

    score(d) = sum over runs of 1 / (k + rank of d in that run)

Only each run's top `depth` documents count; a document a run did not return
adds nothing from that run. RRF uses ranks, not scores, so BM25 scores
(unbounded) and cosine similarities (-1..1) never have to be made comparable.
k=60 is the preregistered default (Cormack et al., 2009).

Ranks are taken with the same tie rule as everywhere else (score descending,
then docid descending), so they match the ranks the evaluator saw.

  python -m src.fuse --dataset trec_covid --runs runs/trec_covid/bm25_query.run runs/trec_covid/dense_query.run

Output: runs/<dataset>/rrf_<query set>.run (+ .manifest.json)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import add_config_args, config_from_args, repo_dir
from src.io_utils import Run, manifest_path, rank_docs, read_run, write_manifest, write_run


def rrf(runs: list[Run], k: int = 60, depth: int = 1000) -> Run:
    fused: Run = {}
    for run in runs:
        for qid, scores in run.items():
            target = fused.setdefault(qid, {})
            for rank, (docid, _) in enumerate(rank_docs(scores)[:depth], 1):
                target[docid] = target.get(docid, 0.0) + 1.0 / (k + rank)
    return fused


def fused_name(run_paths: list[Path]) -> str:
    """bm25_query + dense_query -> rrf_query. The inputs must share a query set."""
    query_sets = {p.name.removesuffix(".run").split("_", 1)[-1] for p in run_paths}
    if len(query_sets) != 1:
        raise ValueError(f"Runs use different query sets {sorted(query_sets)}; pass --name")
    return f"rrf_{query_sets.pop()}"


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="runs to fuse (2 or more)")
    parser.add_argument("--name", help="output run name (default: rrf_<query set>)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")
    if len(args.runs) < 2:
        parser.error("pass at least two runs")

    k, depth = cfg["rrf"]["k"], cfg["rrf"]["depth"]
    start = time.perf_counter()
    fused = rrf([read_run(p) for p in args.runs], k=k, depth=depth)
    seconds = time.perf_counter() - start

    name = args.name or fused_name(args.runs)
    out = write_run(fused, repo_dir(cfg, "runs_dir") / cfg["dataset"]["name"] / f"{name}.run", tag=name, depth=depth)
    write_manifest(manifest_path(out), cfg, system="rrf", inputs=[str(p) for p in args.runs],
                   fuse_seconds=round(seconds, 3), ms_per_query=round(seconds * 1000 / max(len(fused), 1), 3))
    print(f"Fused {len(args.runs)} runs over {len(fused)} queries (k={k}, depth={depth})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
