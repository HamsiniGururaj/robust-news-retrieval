"""RQ3: which kinds of rewording, and which query properties, decide whether
lexical (BM25) or dense retrieval copes better.

Each rewrite is one observation (50 topics x 2 rewrites = 100). For each one:

  change[system]   = score(system, rewrite) - score(system, original query)
  dense_advantage  = change[dense] - change[bm25]     (> 0: dense held up better)

Two analyses:
  1. by phenomenon: mean change per system and mean dense advantage per rewrite
     kind, with a bootstrap 95% CI and a sign-flip randomization test. Holm
     correction across the phenomena with at least `min_group` rewrites; smaller
     groups are listed but not tested.
  2. by feature: Spearman rank correlation between each rewrite feature
     (src.variants) and the dense advantage, with a permutation test and Holm
     correction across features.

  python -m src.robustness --dataset trec_covid
  python -m src.robustness --dataset trec_covid --measure "nDCG(judged_only=True)@10"

Outputs: results/<dataset>/rq3_by_phenomenon_<measure>.csv, rq3_features_<measure>.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.config import add_config_args, config_from_args, repo_dir
from src.evaluate import read_per_topic
from src.io_utils import ensure_parent
from src.stats import bootstrap_ci, holm, randomization_p

SYSTEMS = ["bm25", "dense", "rrf"]
FEATURES = ["content_overlap", "max_idf", "mean_idf", "n_words", "length_change", "n_acronyms"]


def load_observations(features_path: Path, results_dir: Path, measure: str) -> list[dict]:
    """One row per rewrite: its features plus each system's change from the original."""
    with open(features_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    original = {s: read_per_topic(results_dir / f"{s}_query.csv") for s in SYSTEMS}
    rewritten = {(s, v): read_per_topic(results_dir / f"{s}_rewrite_{v}.csv") for s in SYSTEMS for v in "ab"}

    observations = []
    for row in rows:
        qid, v = row["qid"], row["variant"]
        obs = {"qid": qid, "variant": v, "phenomenon": row["phenomenon"], "rewrite": row["rewrite"]}
        for feature in FEATURES:
            obs[feature] = float(row[feature])
        for s in SYSTEMS:
            obs[f"change_{s}"] = rewritten[(s, v)][qid][measure] - original[s][qid][measure]
        obs["dense_advantage"] = obs["change_dense"] - obs["change_bm25"]
        observations.append(obs)
    return observations


def by_phenomenon(observations: list[dict], min_group: int = 10, n_resamples: int = 10000,
                  alpha: float = 0.05, seed: int = 42) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for obs in observations:
        groups.setdefault(obs["phenomenon"], []).append(obs)

    rows = []
    for name, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        adv = np.array([o["dense_advantage"] for o in members])
        row = {"phenomenon": name, "n": len(members)}
        for s in SYSTEMS:
            row[f"mean_change_{s}"] = float(np.mean([o[f"change_{s}"] for o in members]))
        row["mean_dense_advantage"] = float(adv.mean())
        row["dense_better"] = int(np.sum(adv > 0))
        row["bm25_better"] = int(np.sum(adv < 0))
        if len(members) >= min_group:
            rng = np.random.default_rng(seed)
            row["ci_low"], row["ci_high"] = bootstrap_ci(adv, n_resamples, alpha, rng)
            row["p_value"] = randomization_p(adv, n_resamples, rng)
        else:
            row["ci_low"] = row["ci_high"] = row["p_value"] = None
        rows.append(row)

    tested = [r for r in rows if r["p_value"] is not None]
    for r, p in zip(tested, holm([r["p_value"] for r in tested])):
        r["p_holm"] = p
    for r in rows:
        r.setdefault("p_holm", None)
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (average ranks for ties)."""
    def ranks(a):
        order = a.argsort(kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        for value in np.unique(a):  # average ranks of ties
            mask = a == value
            r[mask] = r[mask].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x: np.ndarray, y: np.ndarray, n_resamples: int, rng: np.random.Generator) -> float:
    observed = abs(spearman(x, y))
    extreme = sum(abs(spearman(x, rng.permutation(y))) >= observed - 1e-12 for _ in range(n_resamples))
    return (extreme + 1) / (n_resamples + 1)


def by_feature(observations: list[dict], n_resamples: int = 2000, seed: int = 42) -> list[dict]:
    adv = np.array([o["dense_advantage"] for o in observations])
    rows = []
    for feature in FEATURES:
        x = np.array([o[feature] for o in observations])
        rng = np.random.default_rng(seed)
        rows.append({"feature": feature, "spearman_rho": spearman(x, adv),
                     "p_value": permutation_p(x, adv, n_resamples, rng)})
    for r, p in zip(rows, holm([r["p_value"] for r in rows])):
        r["p_holm"] = p
    return rows


def write_rows(rows: list[dict], path: Path) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else ("" if v is None else v))
                             for k, v in row.items()})
    return path


def _fmt(v, spec="+.3f"):
    return "    -" if v is None else format(v, spec)


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--measure", default="nDCG@10")
    parser.add_argument("--min-group", type=int, default=10, help="smallest phenomenon group that is tested")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")

    name = cfg["dataset"]["name"]
    results_dir = repo_dir(cfg, "results_dir") / name
    features_path = repo_dir(cfg, "queries_dir") / "variants" / f"{name}_rewrite_features.csv"
    observations = load_observations(features_path, results_dir, args.measure)
    n_resamples, alpha, seed = cfg["stats"]["n_resamples"], cfg["stats"]["alpha"], cfg["seed"]

    phen = by_phenomenon(observations, args.min_group, n_resamples, alpha, seed)
    feats = by_feature(observations, seed=seed)
    safe = args.measure.replace("(", "").replace(")", "").replace("=", "").replace("@", "_at_")
    write_rows(phen, results_dir / f"rq3_by_phenomenon_{safe}.csv")
    write_rows(feats, results_dir / f"rq3_features_{safe}.csv")

    print(f"{args.measure}: change from original to rewrite, {len(observations)} rewrites\n")
    print(f"{'phenomenon':<20}{'n':>4}{'BM25':>8}{'dense':>8}{'RRF':>8}{'dense adv':>11}{'95% CI':>18}{'p_holm':>8}  dense/bm25 better")
    for r in phen:
        ci = "" if r["ci_low"] is None else f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"{r['phenomenon']:<20}{r['n']:>4}{r['mean_change_bm25']:>+8.3f}{r['mean_change_dense']:>+8.3f}"
              f"{r['mean_change_rrf']:>+8.3f}{r['mean_dense_advantage']:>+11.3f}{ci:>18}{_fmt(r['p_holm'], '.4f'):>8}"
              f"  {r['dense_better']}/{r['bm25_better']}")
    print(f"\n{'feature':<18}{'Spearman rho':>14}{'p':>9}{'p_holm':>9}   (correlation with dense advantage)")
    for r in feats:
        print(f"{r['feature']:<18}{r['spearman_rho']:>+14.3f}{r['p_value']:>9.4f}{r['p_holm']:>9.4f}")


if __name__ == "__main__":
    main()
