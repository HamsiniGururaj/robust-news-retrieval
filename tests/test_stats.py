"""Significance tests on synthetic data where the right answer is known."""
import numpy as np
import pytest

from src.stats import bootstrap_ci, compare, holm, load_scores, main, paired_values, randomization_p
from src.evaluate import write_per_topic


def per_topic(values, measure="nDCG@10"):
    return {str(i): {measure: float(v)} for i, v in enumerate(values)}


RNG = np.random.default_rng(0)
BASE = RNG.uniform(0.2, 0.8, size=50)


def test_identical_systems_are_not_different():
    [r] = compare({"a": per_topic(BASE), "b": per_topic(BASE)}, [("a", "b")], "nDCG@10", n_resamples=2000)
    assert r.mean_diff == 0
    assert r.p_value == pytest.approx(1.0)
    assert (r.ci_low, r.ci_high) == (0.0, 0.0)
    assert (r.wins, r.ties, r.losses) == (0, 50, 0)


def test_consistent_improvement_is_significant():
    [r] = compare({"a": per_topic(BASE + 0.1), "b": per_topic(BASE)}, [("a", "b")], "nDCG@10", n_resamples=2000)
    assert r.mean_diff == pytest.approx(0.1)
    assert r.p_value < 0.001
    assert r.ci_low == pytest.approx(0.1) and r.ci_high == pytest.approx(0.1)
    assert r.wins == 50


def test_noise_only_difference_is_not_significant():
    noise = np.random.default_rng(1).normal(0, 0.05, size=50)
    noise -= noise.mean()  # zero mean difference by construction
    [r] = compare({"a": per_topic(BASE + noise), "b": per_topic(BASE)}, [("a", "b")], "nDCG@10", n_resamples=2000)
    assert r.p_value > 0.5
    assert r.ci_low < 0 < r.ci_high


def test_randomization_p_matches_exact_value():
    # 10 topics all +0.3: only the all-plus and all-minus sign patterns are as
    # extreme, so the exact two-sided p-value is 2 / 2**10.
    p = randomization_p(np.full(10, 0.3), 50000, np.random.default_rng(0))
    assert p == pytest.approx(2 / 1024, abs=0.0005)


def test_randomization_p_never_zero():
    assert randomization_p(np.full(50, 0.3), 100, np.random.default_rng(0)) == pytest.approx(1 / 101)


def test_bootstrap_ci_contains_mean():
    diff = np.random.default_rng(2).normal(0.05, 0.1, size=50)
    low, high = bootstrap_ci(diff, 2000, 0.05, np.random.default_rng(3))
    assert low < diff.mean() < high


def test_holm_by_hand():
    # sorted: 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_difference_names_compare_changes():
    runs = {
        "bm25_q": per_topic([0.6, 0.6]), "bm25_r": per_topic([0.2, 0.4]),   # drops -0.4, -0.2
        "dense_q": per_topic([0.5, 0.5]), "dense_r": per_topic([0.4, 0.5]),  # drops -0.1,  0.0
    }
    scores = load_scores(["dense_r-dense_q", "bm25_r-bm25_q"], runs.__getitem__)
    assert scores["bm25_r-bm25_q"]["0"]["nDCG@10"] == pytest.approx(-0.4)
    [r] = compare(scores, [("dense_r-dense_q", "bm25_r-bm25_q")], "nDCG@10", n_resamples=500)
    assert r.mean_diff == pytest.approx(0.25)  # dense loses 0.25 less on average
    assert r.wins == 2


def test_mismatched_topics_rejected():
    with pytest.raises(ValueError, match="different topics"):
        paired_values(per_topic([0.1, 0.2]), per_topic([0.1]), "nDCG@10")


def test_command_line(tmp_path, capsys):
    results = tmp_path / "results" / "toy"
    for name, values in {"bm25_q": BASE, "dense_q": BASE + 0.05, "rrf_q": BASE + 0.1}.items():
        write_per_topic({str(i): {"nDCG@10": float(v)} for i, v in enumerate(values)}, name, results / f"{name}.csv")
    main(["--dataset", "scifact", "--set", f"paths.results_dir={(tmp_path / 'results').as_posix()}",
          "--set", "dataset.name=toy", "--set", "stats.n_resamples=500",
          "--runs", "bm25_q", "dense_q", "rrf_q"])
    out = capsys.readouterr().out
    assert "bm25_q vs dense_q" in out and "dense_q vs rrf_q" in out
    assert (results / "significance_bm25_q-dense_q-rrf_q_nDCG_at_10.csv").exists()

    main(["--dataset", "scifact", "--set", f"paths.results_dir={(tmp_path / 'results').as_posix()}",
          "--set", "dataset.name=toy", "--set", "stats.n_resamples=500",
          "--label", "chosen", "--pairs", "rrf_q:bm25_q"])
    out = capsys.readouterr().out
    assert "rrf_q vs bm25_q" in out and "dense_q" not in out
    assert (results / "significance_chosen_nDCG_at_10.csv").exists()
