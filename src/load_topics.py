"""Export a dataset's topics and qrels from ir_datasets into queries/.

For a dataset named <name> this writes:
  queries/<name>_<field>.tsv     one query file per field in dataset.query_fields
                                 (title, desc, ...), the input every retriever reads
  queries/<name>_topics.jsonl    every field of every topic, incl. the narrative, for
                                 the team writing rewrites (they need the full intent)
  queries/<name>.qrels           relevance judgements in TREC format
  queries/<name>.qrels.manifest.json

Only topics and qrels are downloaded; the document corpus is not touched.
ir_datasets caches downloads in ~/.ir_datasets (override with IR_DATASETS_HOME).

  python -m src.load_topics --dataset core18
  python -m src.load_topics --dataset scifact
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.config import add_config_args, config_from_args, repo_dir
from src.io_utils import (
    Qrels, Queries, ensure_parent, manifest_path, qid_sort_key,
    write_manifest, write_qrels, write_queries,
)

# Same threshold the evaluation uses for MAP and Recall (configs/default.yaml).
MIN_RELEVANT = 1

# Short names used in file names, e.g. core18_desc.tsv.
FIELD_FILE_NAMES = {
    "title": "title", "description": "desc", "narrative": "narr", "text": "text", "query": "query",
}


def load_queries(dataset, fields: list[str]) -> dict[str, Queries]:
    """{field: {qid: text}} for each requested field of the dataset's topics."""
    available = dataset.queries_cls()._fields
    missing = [f for f in fields if f not in available]
    if missing:
        raise ValueError(f"Query fields {missing} not in this dataset; available: {list(available)}")

    by_field: dict[str, Queries] = {f: {} for f in fields}
    for query in dataset.queries_iter():
        if query.query_id in by_field[fields[0]]:
            raise ValueError(f"Duplicate query id {query.query_id!r}")
        for field in fields:
            by_field[field][query.query_id] = getattr(query, field)
    return by_field


def load_all_topic_fields(dataset) -> list[dict]:
    """Every field of every topic, as plain dicts (for the rewrite team)."""
    return [dict(query._asdict()) for query in dataset.queries_iter()]


def load_qrels(dataset) -> tuple[Qrels, int]:
    """Qrels plus the number of negative labels that were mapped to 0.

    A few collections carry negative labels (TREC-COVID in BEIR has two -1s).
    They mean "not relevant", but a negative gain would distort nDCG, so they
    are stored as 0.
    """
    qrels: Qrels = {}
    n_negative = 0
    for qrel in dataset.qrels_iter():
        docs = qrels.setdefault(qrel.query_id, {})
        relevance = qrel.relevance
        if relevance < 0:
            n_negative += 1
            relevance = 0
        previous = docs.get(qrel.doc_id)
        if previous is not None and previous != relevance:
            raise ValueError(
                f"Conflicting judgements for qid {qrel.query_id!r}, doc {qrel.doc_id!r}: "
                f"{previous} vs {relevance}"
            )
        docs[qrel.doc_id] = relevance
    return qrels, n_negative


def summarize(queries: Queries, qrels: Qrels) -> dict:
    """Counts used to sanity-check an export before anyone relies on it."""
    relevant_per_topic = {
        qid: sum(rel >= MIN_RELEVANT for rel in qrels.get(qid, {}).values()) for qid in queries
    }
    judged_per_topic = [len(qrels.get(qid, {})) for qid in queries]
    label_counts = Counter(rel for docs in qrels.values() for rel in docs.values())
    rel_counts = list(relevant_per_topic.values())
    return {
        "n_queries": len(queries),
        "n_judgements": sum(len(docs) for docs in qrels.values()),
        "label_counts": {str(label): label_counts[label] for label in sorted(label_counts)},
        "judged_per_topic": _spread(judged_per_topic),
        "relevant_per_topic": _spread(rel_counts),
        # Problems worth a look. A topic with no relevant docs scores 0 for every system.
        "topics_without_relevant": sorted(
            (qid for qid, n in relevant_per_topic.items() if n == 0), key=qid_sort_key
        ),
        "queries_without_qrels": sorted(set(queries) - set(qrels), key=qid_sort_key),
        "qrels_without_queries": sorted(set(qrels) - set(queries), key=qid_sort_key),
    }


def _spread(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def export(dataset, name: str, fields: list[str], out_dir: Path, cfg: dict | None = None) -> dict:
    """Write query files, a full-topics JSONL and qrels for one dataset. Returns the summary."""
    by_field = load_queries(dataset, fields)
    qrels, n_negative = load_qrels(dataset)

    for field, queries in by_field.items():
        write_queries(queries, out_dir / f"{name}_{FIELD_FILE_NAMES.get(field, field)}.tsv")

    topics_path = ensure_parent(out_dir / f"{name}_topics.jsonl")
    with open(topics_path, "w", encoding="utf-8", newline="\n") as f:
        for topic in sorted(load_all_topic_fields(dataset), key=lambda t: qid_sort_key(t["query_id"])):
            f.write(json.dumps(topic, ensure_ascii=False) + "\n")

    qrels_path = write_qrels(qrels, out_dir / f"{name}.qrels")
    summary = summarize(by_field[fields[0]], qrels)
    summary["negative_labels_mapped_to_0"] = n_negative
    write_manifest(manifest_path(qrels_path), cfg, summary=summary)
    return summary


def print_summary(name: str, summary: dict) -> None:
    print(f"{name}: {summary['n_queries']} queries, {summary['n_judgements']} judgements")
    print(f"  labels: {summary['label_counts']}")
    print(f"  judged per topic:   {summary['judged_per_topic']}")
    print(f"  relevant per topic: {summary['relevant_per_topic']}  (rel >= {MIN_RELEVANT})")
    if summary.get("negative_labels_mapped_to_0"):
        print(f"  note: {summary['negative_labels_mapped_to_0']} negative labels stored as 0")
    for key in ("topics_without_relevant", "queries_without_qrels", "qrels_without_queries"):
        if summary[key]:
            print(f"  WARNING {key}: {summary[key]}")


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--out-dir", type=Path, help="default: the repo's queries/ folder")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    ds_cfg = cfg.get("dataset")
    if not ds_cfg:
        parser.error("--dataset is required")

    import ir_datasets  # imported here so the helpers above can be tested without it

    dataset = ir_datasets.load(ds_cfg["queries_id"])
    out_dir = args.out_dir or repo_dir(cfg, "queries_dir")
    summary = export(dataset, ds_cfg["name"], ds_cfg["query_fields"], out_dir, cfg)
    print_summary(ds_cfg["name"], summary)
    print(f"Wrote files to {out_dir}")


if __name__ == "__main__":
    main()
