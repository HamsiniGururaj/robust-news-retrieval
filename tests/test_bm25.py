"""BM25 logic that doesn't need Java. Indexing and searching themselves are
checked in Colab (the index is verified automatically after it is built)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bm25 import (
    check_analysis, default_queries_path, dir_size_mb, hits_to_run, index_command,
    latency_summary, query_set_name, search_queries,
)
from src.config import load_config


def test_query_set_name():
    assert query_set_name(Path("queries/trec_covid_query.tsv"), "trec_covid") == "query"
    assert query_set_name(Path("queries/trec_covid_rewrite_a.tsv"), "trec_covid") == "rewrite_a"


def test_default_queries_are_the_original_queries():
    assert default_queries_path(load_config("trec_covid")).name == "trec_covid_query.tsv"
    assert default_queries_path(load_config("scifact")).name == "scifact_text.tsv"


def test_index_command_disables_stemming_and_keeps_stopwords():
    cmd = index_command(Path("corpus"), Path("index"), load_config()["bm25"])
    assert cmd[cmd.index("--stemmer") + 1] == "none"
    assert "--keepStopwords" in cmd
    assert cmd[cmd.index("--collection") + 1] == "JsonCollection"


def test_check_analysis_accepts_correct_index():
    df = {"the": 900, "studies": 40, "studi": 0}
    assert check_analysis(df.get) == df


@pytest.mark.parametrize("df, problem", [
    ({"the": 0, "studies": 40, "studi": 0}, "stopwords were removed"),
    ({"the": 900, "studies": 0, "studi": 55}, "stemming is on"),
])
def test_check_analysis_rejects_wrong_index(df, problem):
    with pytest.raises(RuntimeError, match=problem):
        check_analysis(df.get)


def test_hits_to_run():
    hits = {"1": [SimpleNamespace(docid="a", score=2.5), SimpleNamespace(docid="b", score=1.0)]}
    assert hits_to_run(hits) == {"1": {"a": 2.5, "b": 1.0}}


def test_search_queries_with_fake_searcher():
    class FakeSearcher:
        def search(self, text, k):
            return [SimpleNamespace(docid=f"{text}-{i}", score=float(k - i)) for i in range(2)]

    run, latencies = search_queries(FakeSearcher(), {"1": "x", "2": "y"}, depth=10)
    assert run == {"1": {"x-0": 10.0, "x-1": 9.0}, "2": {"y-0": 10.0, "y-1": 9.0}}
    assert len(latencies) == 2


def test_latency_summary():
    s = latency_summary([float(i) for i in range(1, 21)])
    assert s == {"n_queries": 20, "mean_ms": 10.5, "median_ms": 10.5, "p95_ms": 20.0}


def test_dir_size_mb(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f").write_bytes(b"x" * 2**20)
    assert dir_size_mb(tmp_path) == 1.0
