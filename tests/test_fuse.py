"""RRF checked against scores worked out by hand (k=60)."""
from pathlib import Path

import pytest

from src.fuse import fused_name, rrf
from src.io_utils import rank_docs

RUN_A = {"1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}, "2": {"x": 1.0, "y": 1.0}}
RUN_B = {"1": {"d3": 0.9, "d1": 0.5}, "3": {"z": 0.2}}


def test_rrf_scores_by_hand():
    fused = rrf([RUN_A, RUN_B], k=60)
    # d1: rank 1 in A, rank 2 in B.  d3: rank 3 in A, rank 1 in B.  d2: rank 2 in A only.
    assert fused["1"]["d1"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["1"]["d3"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["1"]["d2"] == pytest.approx(1 / 62)
    assert [d for d, _ in rank_docs(fused["1"])] == ["d1", "d3", "d2"]


def test_ties_inside_a_run_use_the_shared_tie_rule():
    # x and y tie in run A; docid descending puts y at rank 1.
    fused = rrf([RUN_A], k=60)
    assert fused["2"]["y"] == pytest.approx(1 / 61)
    assert fused["2"]["x"] == pytest.approx(1 / 62)


def test_query_in_only_one_run_is_kept():
    assert rrf([RUN_A, RUN_B])["3"] == {"z": pytest.approx(1 / 61)}


def test_depth_limits_each_input():
    fused = rrf([RUN_A, RUN_B], k=60, depth=1)
    assert fused["1"] == {"d1": pytest.approx(1 / 61), "d3": pytest.approx(1 / 61)}


def test_fused_name():
    assert fused_name([Path("runs/x/bm25_query.run"), Path("runs/x/dense_query.run")]) == "rrf_query"
    assert fused_name([Path("bm25_rewrite_a.run"), Path("dense_rewrite_a.run")]) == "rrf_rewrite_a"
    with pytest.raises(ValueError):
        fused_name([Path("bm25_query.run"), Path("dense_text.run")])
