"""Read and write the shared file formats.

  queries   TSV, one "qid<TAB>text" per line, no header
  qrels     TREC qrels: "qid 0 docid rel"
  runs      TREC run:   "qid Q0 docid rank score tag"
  manifest  JSON written next to a run: config, git commit, environment, timings

In memory, qrels are {qid: {docid: rel}} and runs are {qid: {docid: score}},
the same shapes ir_measures and pytrec_eval use. All qids and docids are strings.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT

Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def qid_sort_key(qid: str) -> tuple:
    """Numeric qids in numeric order ("2" before "10"), then any others alphabetically."""
    return (0, int(qid), "") if qid.isdigit() else (1, 0, qid)


def rank_docs(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Order (docid, score) pairs best first.

    Score ties are broken by docid in descending order, which is what trec_eval
    does, so ranks written here match the ranks the evaluator sees.
    """
    ranked = sorted(scores.items(), key=lambda item: item[0], reverse=True)
    ranked.sort(key=lambda item: item[1], reverse=True)  # stable: keeps the docid order on ties
    return ranked


# ---------------------------------------------------------------- queries

def read_queries(path: str | Path) -> Queries:
    queries: Queries = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            qid, sep, text = line.partition("\t")
            if not sep:
                raise ValueError(f"{path}:{lineno}: expected 'qid<TAB>text'")
            if qid in queries:
                raise ValueError(f"{path}:{lineno}: duplicate qid {qid!r}")
            queries[qid] = text
    return queries


def write_queries(queries: Queries, path: str | Path) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(queries, key=qid_sort_key):
            text = " ".join(queries[qid].split())  # tabs/newlines would break the TSV
            if not text:
                raise ValueError(f"Query {qid!r} is empty")
            f.write(f"{qid}\t{text}\n")
    return path


# ---------------------------------------------------------------- qrels

def read_qrels(path: str | Path) -> Qrels:
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: expected 'qid iter docid rel'")
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def write_qrels(qrels: Qrels, path: str | Path) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(qrels, key=qid_sort_key):
            for docid in sorted(qrels[qid]):
                f.write(f"{qid} 0 {docid} {qrels[qid][docid]}\n")
    return path


# ---------------------------------------------------------------- runs

def read_run(path: str | Path) -> Run:
    run: Run = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"{path}:{lineno}: expected 'qid Q0 docid rank score tag'")
            qid, _, docid, _, score, _ = parts
            docs = run.setdefault(qid, {})
            if docid in docs:
                raise ValueError(f"{path}:{lineno}: docid {docid!r} appears twice for qid {qid!r}")
            docs[docid] = float(score)
    return run


def write_run(run: Run, path: str | Path, tag: str, depth: int | None = None) -> Path:
    """Write a run in TREC format, keeping at most `depth` documents per query."""
    if not tag or any(c.isspace() for c in tag):
        raise ValueError(f"Run tag must be non-empty with no whitespace, got {tag!r}")
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(run, key=qid_sort_key):
            ranked = rank_docs(run[qid])
            if depth is not None:
                ranked = ranked[:depth]
            for rank, (docid, score) in enumerate(ranked, 1):
                # 10 significant digits: enough that rounding doesn't create new ties
                f.write(f"{qid} Q0 {docid} {rank} {score:.10g} {tag}\n")
    return path


# ---------------------------------------------------------------- manifests

def manifest_path(run_path: str | Path) -> Path:
    run_path = Path(run_path)
    return run_path.with_name(run_path.name + ".manifest.json")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def write_manifest(path: str | Path, cfg: dict | None = None, **info: Any) -> Path:
    """Record how an output was produced. Extra keyword args (timings, model revision, ...) are stored as-is."""
    status = _git("status", "--porcelain")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg,
        **info,
    }
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")
    return path
