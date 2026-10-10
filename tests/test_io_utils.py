import json

import pytest

from src.io_utils import (
    manifest_path, rank_docs, read_qrels, read_queries, read_run,
    write_manifest, write_qrels, write_queries, write_run,
)


def test_queries_round_trip_and_cleaning(tmp_path):
    path = write_queries(
        {"10": "second  query", "2": "first\tquery\nwith newline", "abc": "text"},
        tmp_path / "sub" / "q.tsv",  # parent folder is created
    )
    assert read_queries(path) == {"2": "first query with newline", "10": "second query", "abc": "text"}
    assert [line.split("\t")[0] for line in path.read_text(encoding="utf-8").splitlines()] == ["2", "10", "abc"]


def test_empty_query_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_queries({"1": "   "}, tmp_path / "q.tsv")


def test_duplicate_query_rejected(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("1\ta\n1\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_queries(path)


def test_qrels_round_trip(tmp_path):
    qrels = {"321": {"d1": 2, "d2": 0, "d3": 1}, "45": {"d9": 1}}
    assert read_qrels(write_qrels(qrels, tmp_path / "qrels.txt")) == qrels


def test_rank_docs_breaks_ties_by_docid_descending():
    assert rank_docs({"a": 1.0, "c": 2.0, "b": 1.0, "d": 1.0}) == [
        ("c", 2.0), ("d", 1.0), ("b", 1.0), ("a", 1.0),
    ]


def test_run_round_trip_with_ranks(tmp_path):
    run = {"2": {"x": 0.5, "y": 1.5, "z": 1.0}, "1": {"a": 1 / 61}}
    path = write_run(run, tmp_path / "r.run", tag="bm25")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1 Q0 a 1 0.01639344262 bm25"
    assert [line.split()[2:4] for line in lines[1:]] == [["y", "1"], ["z", "2"], ["x", "3"]]
    back = read_run(path)
    assert back.keys() == run.keys()
    assert back["2"] == run["2"]
    assert back["1"]["a"] == pytest.approx(1 / 61)


def test_run_depth_truncates(tmp_path):
    run = {"1": {f"d{i}": float(i) for i in range(10)}}
    back = read_run(write_run(run, tmp_path / "r.run", tag="t", depth=3))
    assert set(back["1"]) == {"d9", "d8", "d7"}


def test_run_tag_without_whitespace(tmp_path):
    with pytest.raises(ValueError):
        write_run({"1": {"d": 1.0}}, tmp_path / "r.run", tag="two words")


@pytest.mark.parametrize("content, match", [
    ("1 Q0 d1 1 0.5\n", "expected"),
    ("1 Q0 d1 1 0.5 t\n1 Q0 d1 2 0.4 t\n", "twice"),
])
def test_malformed_run_rejected(tmp_path, content, match):
    path = tmp_path / "bad.run"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_run(path)


def test_files_use_unix_line_endings(tmp_path):
    path = write_run({"1": {"d": 1.0}}, tmp_path / "r.run", tag="t")
    assert b"\r\n" not in path.read_bytes()


def test_manifest(tmp_path):
    path = write_manifest(manifest_path(tmp_path / "runs" / "bm25.run"), {"bm25": {"k1": 0.9}}, seconds=1.5)
    assert path.name == "bm25.run.manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"] == {"bm25": {"k1": 0.9}}
    assert data["seconds"] == 1.5
    for key in ("created_utc", "git_commit", "git_dirty", "python", "platform"):
        assert key in data
