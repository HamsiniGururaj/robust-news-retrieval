"""Tests use a tiny fake dataset shaped like ir_datasets, so nothing is downloaded."""
import json
from collections import namedtuple

import pytest

from src.io_utils import read_qrels, read_queries
from src.load_topics import export, load_qrels, load_queries, summarize

Topic = namedtuple("Topic", ["query_id", "title", "description", "narrative"])
Qrel = namedtuple("Qrel", ["query_id", "doc_id", "relevance", "iteration"])


class FakeDataset:
    def __init__(self, topics, qrels):
        self.topics, self.qrels = topics, qrels

    def queries_cls(self):
        return Topic

    def queries_iter(self):
        return iter(self.topics)

    def qrels_iter(self):
        return iter(self.qrels)


TOPICS = [
    Topic("10", "Second topic", "Find docs about B.", "Relevant docs discuss B."),
    Topic("2", "First topic", "Find docs about A.", "Relevant docs discuss A."),
]
QRELS = [
    Qrel("2", "d1", 2, "0"), Qrel("2", "d2", 0, "0"), Qrel("2", "d3", 1, "0"),
    Qrel("10", "d4", 0, "0"),  # topic 10 has no relevant docs
    Qrel("99", "d5", 1, "0"),  # judged topic with no query
]


@pytest.fixture
def dataset():
    return FakeDataset(TOPICS, QRELS)


def test_load_queries_by_field(dataset):
    by_field = load_queries(dataset, ["title", "description"])
    assert by_field["title"] == {"10": "Second topic", "2": "First topic"}
    assert by_field["description"]["2"] == "Find docs about A."


def test_unknown_field_rejected(dataset):
    with pytest.raises(ValueError, match="available"):
        load_queries(dataset, ["text"])


def test_duplicate_query_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        load_queries(FakeDataset(TOPICS + [TOPICS[0]], []), ["title"])


def test_load_qrels(dataset):
    assert load_qrels(dataset)["2"] == {"d1": 2, "d2": 0, "d3": 1}


def test_conflicting_qrels_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        load_qrels(FakeDataset([], [Qrel("1", "d", 1, "0"), Qrel("1", "d", 2, "0")]))


def test_repeated_identical_qrel_is_fine():
    assert load_qrels(FakeDataset([], [Qrel("1", "d", 1, "0"), Qrel("1", "d", 1, "0")])) == {"1": {"d": 1}}


def test_summary_flags_problems(dataset):
    s = summarize(load_queries(dataset, ["title"])["title"], load_qrels(dataset))
    assert s["n_queries"] == 2
    assert s["n_judgements"] == 5
    assert s["label_counts"] == {"0": 2, "1": 2, "2": 1}
    assert s["relevant_per_topic"] == {"min": 0, "median": 1.0, "max": 2}
    assert s["topics_without_relevant"] == ["10"]
    assert s["queries_without_qrels"] == []
    assert s["qrels_without_queries"] == ["99"]


def test_export_writes_all_files(dataset, tmp_path):
    export(dataset, "toy", ["title", "description"], tmp_path)

    assert read_queries(tmp_path / "toy_title.tsv") == {"2": "First topic", "10": "Second topic"}
    assert read_queries(tmp_path / "toy_desc.tsv")["10"] == "Find docs about B."
    assert read_qrels(tmp_path / "toy.qrels")["2"] == {"d1": 2, "d2": 0, "d3": 1}

    topics = [json.loads(line) for line in (tmp_path / "toy_topics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [t["query_id"] for t in topics] == ["2", "10"]
    assert topics[0]["narrative"] == "Relevant docs discuss A."

    manifest = json.loads((tmp_path / "toy.qrels.manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["topics_without_relevant"] == ["10"]
