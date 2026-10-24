"""RQ3 analysis on synthetic data with known answers."""
import csv

import numpy as np
import pytest

from src.evaluate import write_per_topic
from src.robustness import FEATURES, by_feature, by_phenomenon, load_observations, spearman


def test_spearman():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x * 10) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, 1.0, 2.0]), np.array([3.0, 3.0, 5.0])) == pytest.approx(1.0)  # ties
    assert spearman(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 0.0  # constant input


def obs(phenomenon, adv, overlap=0.0):
    return {"phenomenon": phenomenon, "dense_advantage": adv, "change_bm25": -adv, "change_dense": 0.0,
            "change_rrf": -adv / 2, **{f: 0.0 for f in FEATURES}, "content_overlap": overlap}


def test_by_phenomenon_tests_only_large_groups():
    data = [obs("paraphrase", 0.2) for _ in range(12)] + [obs("temporal", 0.5)]
    rows = {r["phenomenon"]: r for r in by_phenomenon(data, min_group=10, n_resamples=500)}
    para = rows["paraphrase"]
    assert para["n"] == 12 and para["mean_dense_advantage"] == pytest.approx(0.2)
    assert para["mean_change_bm25"] == pytest.approx(-0.2) and para["mean_change_rrf"] == pytest.approx(-0.1)
    assert para["dense_better"] == 12 and para["p_holm"] < 0.01
    assert rows["temporal"]["p_value"] is None and rows["temporal"]["p_holm"] is None


def test_by_feature_finds_planted_correlation():
    rng = np.random.default_rng(0)
    overlaps = rng.uniform(0, 1, 60)
    data = [obs("paraphrase", -o + rng.normal(0, 0.05), overlap=o) for o in overlaps]  # more overlap -> less dense advantage
    rows = {r["feature"]: r for r in by_feature(data, n_resamples=300)}
    assert rows["content_overlap"]["spearman_rho"] < -0.9
    assert rows["content_overlap"]["p_holm"] < 0.05
    assert rows["n_words"]["spearman_rho"] == 0.0  # constant feature


def test_load_observations(tmp_path):
    results = tmp_path / "results"
    scores = {"bm25_query": 0.6, "dense_query": 0.5, "rrf_query": 0.7,
              "bm25_rewrite_a": 0.2, "dense_rewrite_a": 0.4, "rrf_rewrite_a": 0.5}
    for name, value in scores.items():
        write_per_topic({"1": {"nDCG@10": value}}, name, results / f"{name}.csv")
    for name in ("bm25_rewrite_b", "dense_rewrite_b", "rrf_rewrite_b"):
        write_per_topic({"1": {"nDCG@10": 0.0}}, name, results / f"{name}.csv")
    features = tmp_path / "features.csv"
    with open(features, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "variant", "phenomenon", "rewrite", *FEATURES])
        writer.writeheader()
        writer.writerow({"qid": "1", "variant": "a", "phenomenon": "alias", "rewrite": "x", **{k: 1 for k in FEATURES}})

    [o] = load_observations(features, results, "nDCG@10")
    assert o["change_bm25"] == pytest.approx(-0.4)
    assert o["change_dense"] == pytest.approx(-0.1)
    assert o["dense_advantage"] == pytest.approx(0.3)
