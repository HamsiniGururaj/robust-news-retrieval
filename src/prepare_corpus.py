"""Convert a dataset's documents into one clean JSONL file in Drive.

Both retrievers read this file, so they see exactly the same text:
  BM25 (Pyserini) indexes the "contents" field (title + body)
  dense (E5) encodes title + body as well

Each line is {"id", "title", "body", "contents"}. Whitespace is normalized.
Documents with neither title nor body are skipped (they can't be retrieved);
a repeated document id keeps its first occurrence. Both are counted.

Output: <data_root>/<dataset>/corpus/docs.jsonl
        <data_root>/<dataset>/corpus.manifest.json
The manifest sits outside corpus/ on purpose: Pyserini indexes every .json/.jsonl
file in that folder, and would try to read the manifest as a document.

  python -m src.prepare_corpus --dataset trec_covid
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Iterable

from src.config import add_config_args, config_from_args, data_dir
from src.io_utils import ensure_parent, write_manifest

CORPUS_FILE = "docs.jsonl"


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def make_record(doc_id: str, title: str | None, body: str | None) -> dict | None:
    """One corpus line, or None if the document has no text at all."""
    title, body = clean(title), clean(body)
    if not title and not body:
        return None
    contents = f"{title}\n{body}" if title and body else title or body
    return {"id": doc_id, "title": title, "body": body, "contents": contents}


def convert(docs: Iterable, title_field: str, body_field: str, out_path: Path) -> dict:
    """Write the JSONL and return counts for the manifest.

    Writes to a temporary file first and renames at the end, so an interrupted
    run (e.g. a Colab disconnect) never leaves a half-written corpus behind.
    """
    out_path = ensure_parent(out_path)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    seen: set[str] = set()
    n_written = n_empty = n_duplicate = n_no_body = 0
    lengths: list[int] = []

    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            if doc.doc_id in seen:
                n_duplicate += 1
                continue
            record = make_record(doc.doc_id, getattr(doc, title_field), getattr(doc, body_field))
            if record is None:
                n_empty += 1
                continue
            seen.add(doc.doc_id)
            n_no_body += not record["body"]
            lengths.append(len(record["contents"].split()))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    os.replace(tmp_path, out_path)
    return {
        "n_documents": n_written,
        "n_skipped_empty": n_empty,
        "n_skipped_duplicate_id": n_duplicate,
        "n_title_only": n_no_body,
        # whitespace words, a rough guide to how much E5's 512-token limit truncates
        "words_per_doc": {
            "mean": round(statistics.fmean(lengths), 1) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths, default=0),
            "share_over_400_words": round(sum(n > 400 for n in lengths) / len(lengths), 4) if lengths else 0,
        },
    }


def read_corpus(path: str | Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def corpus_path(cfg: dict) -> Path:
    return data_dir(cfg, "corpus", CORPUS_FILE)


def corpus_manifest_path(cfg: dict) -> Path:
    return data_dir(cfg, "corpus.manifest.json")


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--force", action="store_true", help="rebuild even if the corpus file exists")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    ds = cfg.get("dataset")
    if not ds:
        parser.error("--dataset is required")

    out_path = corpus_path(cfg)
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; use --force to rebuild")
        return

    import ir_datasets  # imported here so the helpers above can be tested without it

    start = time.perf_counter()
    stats = convert(ir_datasets.load(ds["docs_id"]).docs_iter(), ds["title_field"], ds["body_field"], out_path)
    stats["seconds"] = round(time.perf_counter() - start, 1)
    stats["size_mb"] = round(out_path.stat().st_size / 2**20, 1)
    write_manifest(corpus_manifest_path(cfg), cfg, **stats)

    print(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
