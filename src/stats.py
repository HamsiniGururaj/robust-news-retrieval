"""Paired significance tests between runs, over topics.

Is system A really better than system B, or could the difference be luck of
which 50 topics were picked? For every pair of runs, on one measure:

  mean difference   mean over topics of (A - B)           the effect size
  95% CI            paired bootstrap: resample topics with replacement
  p-value           paired randomization test: randomly swap A/B per topic
  Holm-adjusted p   corrects for running several comparisons at once
  wins/ties/losses  topics where A beat, tied or lost to B

Both runs must cover the same topics (always true for runs scored by
src.evaluate against the same qrels).

  python -m src.stats --dataset trec_covid --runs bm25_query dense_query rrf_query
  python -m src.stats --dataset trec_covid --runs bm25_query dense_query rrf_query --measure "AP(rel=1)"
  python -m src.stats --dataset trec_covid --label query_form \
      --pairs bm25_text:bm25_query dense_text:dense_query rrf_text:rrf_query

--runs compares every pair of the listed runs; --pairs compares only the given
A:B pairs. Either way, Holm correction is applied across the comparisons made.

Output: results/<dataset>/significance_<label>.csv (+ a printed table)
"""
from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.config import add_config_args, config_from_args, repo_dir
from src.evaluate import read_per_topic
from src.io_utils import ensure_parent, qid_sort_key


@dataclass
class Comparison:
    run_a: str
    run_b: str
    measure: str
    n_topics: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float
    wins: int
    ties: int
    losses: int


def paired_values(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], measure: str):
    if set(a) != set(b):
        raise ValueError(f"Runs cover different topics ({len(a)} vs {len(b)}); score them against the same qrels")
    qids = sorted(a, key=qid_sort_key)
    return np.array([a[q][measure] for q in qids]), np.array([b[q][measure] for q in qids])


def bootstrap_ci(diff: np.ndarray, n_resamples: int, alpha: float, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI of the mean difference, resampling topics with replacement."""
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def randomization_p(diff: np.ndarray, n_resamples: int, rng: np.random.Generator) -> float:
    """Two-sided p-value: how often randomly swapping A/B per topic gives a mean
    difference at least as large as the observed one. (+1 so p is never 0.)"""
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(diff)))
    extreme = np.sum(np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12)
    return float((extreme + 1) / (n_resamples + 1))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def compare(scores: dict[str, dict], pairs: list[tuple[str, str]], measure: str,
            n_resamples: int = 10000, alpha: float = 0.05, seed: int = 42) -> list[Comparison]:
    """Test every pair; Holm correction is applied across the pairs given."""
    results = []
    for run_a, run_b in pairs:
        a, b = paired_values(scores[run_a], scores[run_b], measure)
        diff = a - b
        rng = np.random.default_rng(seed)  # same seed per pair: results don't depend on pair order
        low, high = bootstrap_ci(diff, n_resamples, alpha, rng)
        results.append(Comparison(
            run_a=run_a, run_b=run_b, measure=measure, n_topics=len(diff),
            mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(diff.mean()),
            ci_low=low, ci_high=high, p_value=randomization_p(diff, n_resamples, rng), p_holm=0.0,
            wins=int(np.sum(diff > 0)), ties=int(np.sum(diff == 0)), losses=int(np.sum(diff < 0)),
        ))
    for result, adjusted in zip(results, holm([r.p_value for r in results])):
        result.p_holm = adjusted
    return results


def write_comparisons(results: list[Comparison], path: str | Path) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])), lineterminator="\n")
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    return path


def format_table(results: list[Comparison], alpha: float) -> str:
    lines = [f"{'A vs B':<28} {'mean A':>7} {'mean B':>7} {'diff':>7} {'95% CI':>17} {'p':>7} {'p_holm':>7}  W/T/L"]
    for r in results:
        mark = " *" if r.p_holm < alpha else ""
        lines.append(
            f"{r.run_a + ' vs ' + r.run_b:<28} {r.mean_a:>7.4f} {r.mean_b:>7.4f} {r.mean_diff:>+7.4f} "
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] {r.p_value:>7.4f} {r.p_holm:>7.4f}  "
            f"{r.wins}/{r.ties}/{r.losses}{mark}"
        )
    lines.append(f"* significant after Holm correction at alpha={alpha}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--runs", nargs="+", help="compare every pair, e.g. bm25_query dense_query rrf_query")
    which.add_argument("--pairs", nargs="+", metavar="A:B", help="compare only these pairs, e.g. bm25_text:bm25_query")
    parser.add_argument("--measure", default="nDCG@10")
    parser.add_argument("--label", help="output name (default: the run names joined with '-')")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")

    if args.pairs:
        pairs = [tuple(p.split(":", 1)) for p in args.pairs]
        if any(len(p) != 2 or not all(p) for p in pairs):
            parser.error("pairs must look like A:B")
    else:
        if len(args.runs) < 2:
            parser.error("pass at least two runs")
        pairs = list(itertools.combinations(args.runs, 2))
    names = list(dict.fromkeys(name for pair in pairs for name in pair))

    results_dir = repo_dir(cfg, "results_dir") / cfg["dataset"]["name"]
    scores = {name: read_per_topic(results_dir / f"{name}.csv") for name in names}
    stats_cfg = cfg["stats"]
    results = compare(scores, pairs, args.measure, stats_cfg["n_resamples"], stats_cfg["alpha"], cfg["seed"])

    label = args.label or "-".join(names)
    safe_measure = args.measure.replace("(", "").replace(")", "").replace("=", "").replace("@", "_at_")
    out = write_comparisons(results, results_dir / f"significance_{label}_{safe_measure}.csv")
    print(f"{args.measure}, {results[0].n_topics} topics, {stats_cfg['n_resamples']} resamples\n")
    print(format_table(results, stats_cfg["alpha"]))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
