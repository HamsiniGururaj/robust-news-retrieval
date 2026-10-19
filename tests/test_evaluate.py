"""Evaluation checked against scores worked out by hand.

Fake qrels (relevance 0/1/2) and a fake run:

  topic 1: judged d1=2, d2=1, d3=0, d4=1  -> 3 relevant docs (d1, d2, d4)
           run ranks d3, d1, d2           -> d4 is never retrieved
  topic 2: judged d5=1
           run gives d0 and d5 the SAME score; trec_eval breaks ties by docid
           descending, so d5 is rank 1
  topic 3: judged d6=1, but the run has nothing for it -> must score 0
  topic 9: in the run but not in the qrels -> ignored
"""
import math
from pathlib import Path

import pytest

from src.evaluate import (
    evaluate_run, expand_paths, main, mean_scores, read_per_topic, run_name, write_per_topic,
)
from src.io_utils import write_qrels, write_run

MEASURES = ["nDCG@10", "AP(rel=1)", "R(rel=1)@100"]
QRELS = {
    "1": {"d1": 2, "d2": 1, "d3": 0, "d4": 1},
    "2": {"d5": 1},
    "3": {"d6": 1},
}
RUN = {
    "1": {"d3": 3.0, "d1": 2.0, "d2": 1.0},
    "2": {"d0": 1.0, "d5": 1.0},
    "9": {"d7": 1.0},
}

# Topic 1 by hand. nDCG gain = relevance label, discount = log2(rank + 1).
DCG_1 = 0 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)   # d3, d1, d2
IDCG_1 = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # ideal: 2, 1, 1
NDCG_1 = DCG_1 / IDCG_1                                          # ~0.5627
AP_1 = (1 / 2 + 2 / 3) / 3   # relevant at ranks 2 and 3, out of 3 relevant -> ~0.3889
RECALL_1 = 2 / 3             # 2 of 3 relevant docs in the top 100


@pytest.fixture
def scores():
    scores, unjudged = evaluate_run(RUN, QRELS, MEASURES)
    assert unjudged == ["9"]
    return scores


def test_topic_scores_match_hand_calculation(scores):
    assert scores["1"]["nDCG@10"] == pytest.approx(NDCG_1)
    assert scores["1"]["AP(rel=1)"] == pytest.approx(AP_1)
    assert scores["1"]["R(rel=1)@100"] == pytest.approx(RECALL_1)


def test_ties_broken_like_io_utils(scores):
    # If ties went the other way, d5 would be rank 2 and AP would be 0.5.
    assert scores["2"] == {"nDCG@10": 1.0, "AP(rel=1)": 1.0, "R(rel=1)@100": 1.0}


def test_missing_topic_scores_zero_and_unjudged_ignored(scores):
    assert scores["3"] == {"nDCG@10": 0.0, "AP(rel=1)": 0.0, "R(rel=1)@100": 0.0}
    assert "9" not in scores


def test_mean_includes_the_zero_topic(scores):
    means = mean_scores(scores, MEASURES)
    assert means["nDCG@10"] == pytest.approx((NDCG_1 + 1 + 0) / 3)
    assert means["AP(rel=1)"] == pytest.approx((AP_1 + 1 + 0) / 3)
    assert means["R(rel=1)@100"] == pytest.approx((RECALL_1 + 1 + 0) / 3)


def test_judged_at_10():
    scores, _ = evaluate_run(RUN, QRELS, ["Judged@10"])
    assert scores["1"]["Judged@10"] == pytest.approx(3 / 3)  # d3, d1, d2 all judged
    assert scores["2"]["Judged@10"] == pytest.approx(1 / 2)  # d5 judged, d0 not
    assert scores["3"]["Judged@10"] == 0.0                   # nothing retrieved


def test_per_topic_csv_round_trip(scores, tmp_path):
    path = write_per_topic(scores, "toy", tmp_path / "toy.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "run,qid,measure,value"
    back = read_per_topic(path)
    assert back.keys() == scores.keys()
    assert back["1"]["nDCG@10"] == pytest.approx(NDCG_1, abs=1e-6)


def test_expand_paths(tmp_path):
    for name in ("a.run", "b.run", "c.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [p.name for p in expand_paths([str(tmp_path / "*.run")])] == ["a.run", "b.run"]
    with pytest.raises(FileNotFoundError):
        expand_paths([str(tmp_path / "*.nothing")])


def test_run_name():
    assert run_name(Path("runs/scifact/bm25_text.run")) == "bm25_text"


def test_command_line(tmp_path, capsys):
    qrels_path = write_qrels(QRELS, tmp_path / "toy.qrels")
    run_path = write_run(RUN, tmp_path / "toy_bm25.run", tag="toy")
    main(["--runs", str(run_path), "--qrels", str(qrels_path), "--out-dir", str(tmp_path / "results")])

    out = capsys.readouterr().out
    assert "toy_bm25" in out and "1 topics not in the qrels" in out
    assert read_per_topic(tmp_path / "results" / "toy_bm25.csv")["3"]["AP(rel=1)"] == 0.0
