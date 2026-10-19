from collections import namedtuple

from src.prepare_corpus import convert, make_record, read_corpus

Doc = namedtuple("Doc", ["doc_id", "title", "text"])


def test_make_record_joins_and_cleans():
    record = make_record("d1", "  A  title ", "Body\ttext\nhere")
    assert record == {"id": "d1", "title": "A title", "body": "Body text here", "contents": "A title\nBody text here"}


def test_title_only_and_body_only():
    assert make_record("d", "Title", "")["contents"] == "Title"
    assert make_record("d", "", "Body")["contents"] == "Body"
    assert make_record("d", None, "Body")["title"] == ""


def test_empty_document_skipped():
    assert make_record("d", "  ", None) is None


def test_convert_counts_and_output(tmp_path):
    docs = [
        Doc("d1", "First", "one two three"),
        Doc("d2", "Second", ""),            # title only
        Doc("d3", "", ""),                  # empty -> skipped
        Doc("d1", "Duplicate", "ignored"),  # repeated id -> skipped
    ]
    out = tmp_path / "corpus" / "docs.jsonl"
    stats = convert(docs, "title", "text", out)

    assert [r["id"] for r in read_corpus(out)] == ["d1", "d2"]
    assert stats["n_documents"] == 2
    assert stats["n_skipped_empty"] == 1
    assert stats["n_skipped_duplicate_id"] == 1
    assert stats["n_title_only"] == 1
    assert stats["words_per_doc"]["max"] == 4  # "First one two three"
    assert not (tmp_path / "corpus" / "docs.jsonl.tmp").exists()
