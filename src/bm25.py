"""BM25 retrieval with Pyserini (Lucene). Needs Java, so it runs in Colab.

Two steps:

  index   Build a Lucene inverted index over corpus/docs.jsonl ("contents" =
          title + body). Stemming is off and stopwords are kept, then the
          index is checked to prove it (see verify_index).
  search  Run a query file against the index with BM25 (k1, b from the config)
          and write the top `depth` documents per query as a TREC run, with
          per-query latency in the run's manifest.

  python -m src.bm25 index  --dataset trec_covid
  python -m src.bm25 search --dataset trec_covid                         # original queries
  python -m src.bm25 search --dataset trec_covid --queries queries/trec_covid_text.tsv

Outputs:
  <data_root>/<dataset>/bm25_index/            the Lucene index (Drive)
  <data_root>/<dataset>/bm25_index.manifest.json
  runs/<dataset>/bm25_<query set>.run (+ .manifest.json)
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.config import add_config_args, config_from_args, data_dir, repo_dir
from src.io_utils import Run, manifest_path, read_queries, write_manifest, write_run
from src.load_topics import default_queries_path, query_set_name

# A word Lucene's English stopword list would remove, and a word whose Porter stem
# ("studi") differs from itself. If stemming/stopword removal were (wrongly) on,
# "the" would be missing from the index and "studi" would be present.
STOPWORD_PROBE = "the"
STEM_PROBE, STEMMED_FORM = "studies", "studi"


# ---------------------------------------------------------------- paths and names

def index_dir(cfg: dict) -> Path:
    return data_dir(cfg, "bm25_index")


def index_manifest_path(cfg: dict) -> Path:
    return data_dir(cfg, "bm25_index.manifest.json")


def dir_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20, 1)


# ---------------------------------------------------------------- indexing

def index_command(corpus_dir: Path, out_dir: Path, bm25_cfg: dict) -> list[str]:
    """The Pyserini indexing command. Positions/docvectors/raw text are stored so
    we can inspect documents and term statistics later (error analysis, IDF features)."""
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(corpus_dir),
        "--index", str(out_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(bm25_cfg["threads"]),
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--stemmer", bm25_cfg["stemmer"],
    ]
    if bm25_cfg["keep_stopwords"]:
        cmd.append("--keepStopwords")
    return cmd


def check_analysis(document_frequency) -> dict:
    """Check the index really has no stemming and keeps stopwords.

    `document_frequency(term)` returns how many documents contain the exact,
    unanalyzed term. Raises if the index was built with the wrong analyzer.
    """
    found = {term: document_frequency(term) for term in (STOPWORD_PROBE, STEM_PROBE, STEMMED_FORM)}
    if found[STOPWORD_PROBE] == 0:
        raise RuntimeError(f"Stopword {STOPWORD_PROBE!r} is missing from the index: stopwords were removed")
    if found[STEMMED_FORM] > 0 and found[STEM_PROBE] == 0:
        raise RuntimeError(f"Index contains {STEMMED_FORM!r} but not {STEM_PROBE!r}: Porter stemming is on")
    return found


def _index_reader(path: Path):
    try:
        from pyserini.index.lucene import LuceneIndexReader
    except ImportError:  # older Pyserini
        from pyserini.index.lucene import IndexReader as LuceneIndexReader
    return LuceneIndexReader(str(path))


def verify_index(path: Path) -> dict:
    reader = _index_reader(path)
    probes = check_analysis(lambda term: reader.get_term_counts(term, analyzer=None)[0])
    return {"probe_document_frequencies": probes, "index_stats": reader.stats()}


def build_index(cfg: dict, force: bool = False) -> dict:
    from src.prepare_corpus import corpus_path

    corpus = corpus_path(cfg)
    out_dir = index_dir(cfg)
    if not corpus.exists():
        raise FileNotFoundError(f"{corpus} not found; run: python -m src.prepare_corpus --dataset {cfg['dataset']['name']}")
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"{out_dir} already exists; use --force to rebuild")
        return json.loads(index_manifest_path(cfg).read_text(encoding="utf-8")) if index_manifest_path(cfg).exists() else {}

    # Build on local disk, then copy: Lucene writes many small files and uses file
    # locks, which the Google Drive mount in Colab handles poorly.
    with tempfile.TemporaryDirectory(prefix="bm25_index_") as tmp:
        local_dir = Path(tmp) / "index"
        cmd = index_command(corpus.parent, local_dir, cfg["bm25"])
        print("Running:", " ".join(cmd))
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        index_seconds = round(time.perf_counter() - start, 1)
        checks = verify_index(local_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(local_dir, out_dir)

    info = {
        "index_seconds": index_seconds,
        "index_size_mb": dir_size_mb(out_dir),
        "command": cmd,
        **checks,
    }
    write_manifest(index_manifest_path(cfg), cfg, **info)
    return info


# ---------------------------------------------------------------- searching

def hits_to_run(results: dict[str, list]) -> Run:
    """Pyserini hits (objects with .docid and .score) -> {qid: {docid: score}}."""
    return {qid: {hit.docid: float(hit.score) for hit in hits} for qid, hits in results.items()}


def make_searcher(path: Path, bm25_cfg: dict):
    from pyserini.analysis import get_lucene_analyzer
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(path))
    searcher.set_bm25(bm25_cfg["k1"], bm25_cfg["b"])
    # Queries must be analyzed like the documents were: no stemming, stopwords kept.
    searcher.set_analyzer(get_lucene_analyzer(
        stemming=bm25_cfg["stemmer"] != "none", stopwords=not bm25_cfg["keep_stopwords"],
    ))
    return searcher


def search_queries(searcher, queries: dict[str, str], depth: int) -> tuple[Run, list[float]]:
    """Search one query at a time so each query's latency can be measured.

    One untimed warm-up query runs first, so Java start-up costs don't count as latency.
    """
    if queries:
        searcher.search(next(iter(queries.values())), k=depth)
    results, latencies_ms = {}, []
    for qid, text in queries.items():
        start = time.perf_counter()
        results[qid] = searcher.search(text, k=depth)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return hits_to_run(results), latencies_ms


def latency_summary(latencies_ms: list[float]) -> dict:
    ordered = sorted(latencies_ms)
    return {
        "n_queries": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
    }


def run_search(cfg: dict, queries_path: Path | None = None) -> Path:
    name = cfg["dataset"]["name"]
    queries_path = queries_path or default_queries_path(cfg)
    queries = read_queries(queries_path)

    # Search a local copy: reading the index over the Drive mount would inflate latency.
    with tempfile.TemporaryDirectory(prefix="bm25_search_") as tmp:
        local_index = shutil.copytree(index_dir(cfg), Path(tmp) / "index")
        searcher = make_searcher(local_index, cfg["bm25"])
        run, latencies = search_queries(searcher, queries, cfg["bm25"]["depth"])
        getattr(searcher, "close", lambda: None)()  # release index files before the temp dir is removed
    run_tag = f"bm25_{query_set_name(queries_path, name)}"
    out = write_run(run, repo_dir(cfg, "runs_dir") / name / f"{run_tag}.run", tag=run_tag, depth=cfg["bm25"]["depth"])

    index_info = {}
    if index_manifest_path(cfg).exists():
        saved = json.loads(index_manifest_path(cfg).read_text(encoding="utf-8"))
        index_info = {k: saved.get(k) for k in ("index_seconds", "index_size_mb", "index_stats")}
    write_manifest(
        manifest_path(out), cfg, system="bm25", queries_file=str(queries_path),
        latency=latency_summary(latencies), index=index_info,
    )
    print(f"{len(run)} queries, latency {latency_summary(latencies)}")
    print(f"Wrote {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_index = add_config_args(sub.add_parser("index", help="build the Lucene index"))
    p_index.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    p_search = add_config_args(sub.add_parser("search", help="search a query file"))
    p_search.add_argument("--queries", type=Path, help="query TSV (default: the dataset's original queries)")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")
    if args.command == "index":
        print(json.dumps(build_index(cfg, args.force), indent=2, default=str))
    else:
        run_search(cfg, args.queries)


if __name__ == "__main__":
    main()
