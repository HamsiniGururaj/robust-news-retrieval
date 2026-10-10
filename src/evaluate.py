"""Score TREC run files against qrels: nDCG@10, MAP and Recall@100, per topic and mean.

A run is the ranked list a system produced; qrels are the human judgements.
For each run this writes results/<dataset>/<run name>.csv with one row per
(topic, measure) and prints the mean of every measure.

Every topic in the qrels is scored. A topic the system returned nothing for
scores 0 (like trec_eval -c), so a system can't look better by skipping hard
topics. Topics in the run but not in the qrels are ignored and reported.

  python -m src.evaluate --dataset scifact --runs runs/scifact/*.run
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from src.config import add_config_args, config_from_args, repo_dir
from src.io_utils import Qrels, Run, ensure_parent, qid_sort_key, read_qrels, read_run

PerTopic = dict[str, dict[str, float]]  # qid -> measure name -> value


def evaluate_run(run: Run, qrels: Qrels, measures: list[str]) -> tuple[PerTopic, list[str]]:
    """Per-topic scores for every qrels topic, plus the run topics that have no qrels."""
    import ir_measures

    # ir_measures reports "AP(rel=1)" as "AP"; map its names back to the config's.
    parsed = {str(ir_measures.parse_measure(name)): name for name in measures}
    evaluator = ir_measures.evaluator([ir_measures.parse_measure(name) for name in measures], qrels)

    scores: PerTopic = {qid: {name: 0.0 for name in measures} for qid in qrels}
    for metric in evaluator.iter_calc(run):
        if metric.query_id in scores:
            scores[metric.query_id][parsed[str(metric.measure)]] = float(metric.value)

    unjudged = sorted(set(run) - set(qrels), key=qid_sort_key)
    return scores, unjudged


def mean_scores(scores: PerTopic, measures: list[str]) -> dict[str, float]:
    if not scores:
        return {name: 0.0 for name in measures}
    return {name: sum(topic[name] for topic in scores.values()) / len(scores) for name in measures}


def write_per_topic(scores: PerTopic, run_name: str, path: str | Path) -> Path:
    """Long format, one row per (topic, measure): easy to filter, join and pair later."""
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["run", "qid", "measure", "value"])
        for qid in sorted(scores, key=qid_sort_key):
            for name, value in scores[qid].items():
                writer.writerow([run_name, qid, name, f"{value:.6f}"])
    return path


def read_per_topic(path: str | Path) -> PerTopic:
    scores: PerTopic = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores.setdefault(row["qid"], {})[row["measure"]] = float(row["value"])
    return scores


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand wildcards ourselves: PowerShell passes "runs/*.run" through unexpanded."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise FileNotFoundError(f"No files match {pattern!r}")
        paths.extend(Path(m) for m in matches)
    return paths


def run_name(path: Path) -> str:
    return path.name.removesuffix(".run")


def format_table(means: dict[str, dict[str, float]], measures: list[str]) -> str:
    width = max([len("run")] + [len(name) for name in means])
    lines = [f"{'run':<{width}}  " + "  ".join(f"{m:>12}" for m in measures)]
    for name, values in means.items():
        lines.append(f"{name:<{width}}  " + "  ".join(f"{values[m]:>12.4f}" for m in measures))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--runs", nargs="+", required=True, help="run files; wildcards allowed")
    parser.add_argument("--qrels", type=Path, help="default: queries/<dataset>.qrels")
    parser.add_argument("--out-dir", type=Path, help="default: results/<dataset>/")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    name = (cfg.get("dataset") or {}).get("name")
    if not name and not (args.qrels and args.out_dir):
        parser.error("pass --dataset, or both --qrels and --out-dir")
    qrels_path = args.qrels or repo_dir(cfg, "queries_dir") / f"{name}.qrels"
    out_dir = args.out_dir or repo_dir(cfg, "results_dir") / name
    measures = cfg["eval"]["measures"]

    qrels = read_qrels(qrels_path)
    print(f"qrels: {qrels_path} ({len(qrels)} topics)")

    means: dict[str, dict[str, float]] = {}
    for path in expand_paths(args.runs):
        scores, unjudged = evaluate_run(read_run(path), qrels, measures)
        out = write_per_topic(scores, run_name(path), out_dir / f"{run_name(path)}.csv")
        means[run_name(path)] = mean_scores(scores, measures)
        if unjudged:
            print(f"  note: {path.name} has {len(unjudged)} topics not in the qrels (ignored)")
        print(f"  {path.name} -> {out}")

    print()
    print(format_table(means, measures))


if __name__ == "__main__":
    main()
