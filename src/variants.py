"""Query rewrites: a fill-in template, automatic checks, and export for the runs.

The rewrites are written by hand in one CSV, one row per topic:

  queries/variants/<dataset>_rewrites.csv
    qid, original, question, narrative, anchor   reference (filled in for you)
    rewrite_a, phenomenon_a                      first rewrite
    rewrite_b, phenomenon_b                      second rewrite
    verified                                     "yes" after the second-pass check
    notes                                        free text

  python -m src.variants template --dataset trec_covid   # create the CSV (never overwrites)
  python -m src.variants check    --dataset trec_covid   # check it; export if there are no errors

`check` reports ERRORS (must fix: missing fields, unknown phenomenon, a
negation word added, ...) and WARNINGS (look again: the anchor word is still
there, high word overlap, a length or question-form change, not yet verified).
With no errors it writes:
  queries/<dataset>_rewrite_a.tsv, queries/<dataset>_rewrite_b.tsv   inputs for the runs
  queries/variants/<dataset>_rewrite_features.csv                  one row per rewrite, for RQ3

The "anchor" is the original query's rarest content word (highest IDF in the
corpus): the word BM25 leans on most. Rewrites should not contain it.
Document frequencies are computed once from the corpus and cached in .cache/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

from src.config import REPO_ROOT, add_config_args, config_from_args, repo_dir
from src.io_utils import ensure_parent, qid_sort_key, read_queries, write_queries
from src.load_topics import default_queries_path

PHENOMENA = {
    "paraphrase": "synonym or paraphrase of a content phrase",
    "alias": "acronym/abbreviation <-> full name, or another established name",
    "entity_description": "a named entity replaced by an unambiguous description",
    "temporal": "an equivalent date or period expression",
    "deanchoring": "the rarest (anchor) term removed and the need restated",
}
# Words that can flip or narrow relevance; not allowed to appear unless the original has them.
NEGATIONS = {"not", "no", "non", "without", "except", "excluding", "never", "nor", "neither", "none", "lack", "absence"}
QUESTION_WORDS = {"how", "what", "why", "when", "where", "which", "who", "does", "do", "is", "are", "can", "will"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "has", "have",
    "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "between", "during", "after", "before", "about", "against", "people",
}
COLUMNS = ["qid", "original", "question", "narrative", "anchor",
           "rewrite_a", "phenomenon_a", "rewrite_b", "phenomenon_b", "verified", "notes"]
MAX_LENGTH_CHANGE = 3     # words
MAX_CONTENT_OVERLAP = 0.5  # Jaccard of content words


# ---------------------------------------------------------------- text features

def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def standalone_numbers(text: str) -> set[str]:
    """Numbers that stand alone ("2020", "65", "3.5"), not parts of names like SARS-CoV-2 or COVID-19."""
    return set(re.findall(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])", text))


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def idf(term: str, df: dict[str, int], n_docs: int) -> float:
    """BM25-style IDF. Unseen terms get the maximum (they are the rarest possible)."""
    d = df.get(term, 0)
    return math.log(1 + (n_docs - d + 0.5) / (d + 0.5))


def anchor_term(text: str, df: dict[str, int], n_docs: int) -> str:
    """The content word with the highest IDF (ties: the longer word)."""
    words = content_words(text) or set(tokens(text))
    return max(sorted(words), key=lambda w: (idf(w, df, n_docs), len(w)))


def features(original: str, rewrite: str, df: dict[str, int], n_docs: int) -> dict:
    orig_cw, rew_cw = content_words(original), content_words(rewrite)
    anchor = anchor_term(original, df, n_docs)
    return {
        # whitespace words, so "SARS-CoV-2" counts as one word
        "n_words": len(rewrite.split()),
        "length_change": len(rewrite.split()) - len(original.split()),
        "content_overlap": round(jaccard(orig_cw, rew_cw), 3),
        "anchor": anchor,
        "anchor_removed": anchor not in set(tokens(rewrite)),
        "max_idf": round(max((idf(w, df, n_docs) for w in rew_cw), default=0.0), 3),
        "mean_idf": round(sum(idf(w, df, n_docs) for w in rew_cw) / len(rew_cw), 3) if rew_cw else 0.0,
        "n_numbers": len(standalone_numbers(rewrite)),
        "n_acronyms": sum(sum(c.isupper() for c in w) >= 2 for w in rewrite.split()),  # ACE, SARS-CoV-2, mRNA
        "n_capitalized": len(re.findall(r"\b[A-Z][a-z]+", rewrite)),
    }


# ---------------------------------------------------------------- checking

def check_rewrite(original: str, rewrite: str, phenomenon: str, label: str, df, n_docs) -> tuple[list, list]:
    errors, warnings = [], []
    if not rewrite.strip():
        return [f"{label}: missing"], []
    if phenomenon not in PHENOMENA:
        errors.append(f"{label}: phenomenon {phenomenon!r} is not one of {sorted(PHENOMENA)}")
    if tokens(rewrite) == tokens(original):
        errors.append(f"{label}: same words as the original")
    orig_t, rew_t = set(tokens(original)), set(tokens(rewrite))
    added_negations = (rew_t & NEGATIONS) - orig_t
    if added_negations:
        errors.append(f"{label}: adds negation {sorted(added_negations)}; this can change relevance")
    if standalone_numbers(rewrite) != standalone_numbers(original):
        warnings.append(f"{label}: numbers differ from the original; make sure the meaning is unchanged")

    f = features(original, rewrite, df, n_docs)
    if not f["anchor_removed"]:
        warnings.append(f"{label}: still contains the anchor word {f['anchor']!r}")
    if f["content_overlap"] > MAX_CONTENT_OVERLAP:
        warnings.append(f"{label}: content-word overlap {f['content_overlap']} > {MAX_CONTENT_OVERLAP}")
    if abs(f["length_change"]) > MAX_LENGTH_CHANGE:
        warnings.append(f"{label}: length changed by {f['length_change']:+d} words (keep within +/-{MAX_LENGTH_CHANGE})")
    starts_question = lambda text: bool(tokens(text)) and tokens(text)[0] in QUESTION_WORDS
    if starts_question(rewrite) != starts_question(original):
        warnings.append(f"{label}: question/keyword form differs from the original")
    return errors, warnings


def check_rows(rows: list[dict], topic_qids: set[str], df, n_docs) -> tuple[dict, dict]:
    """{qid: [errors]}, {qid: [warnings]} for the whole sheet."""
    errors: dict[str, list] = {}
    warnings: dict[str, list] = {}
    seen = Counter(row["qid"] for row in rows)
    for qid in sorted(topic_qids - set(seen), key=qid_sort_key):
        errors.setdefault(qid, []).append("topic missing from the sheet")
    for qid, n in seen.items():
        if n > 1:
            errors.setdefault(qid, []).append(f"appears {n} times")
    for row in rows:
        qid = row["qid"]
        for v in ("a", "b"):
            e, w = check_rewrite(row["original"], row[f"rewrite_{v}"], row[f"phenomenon_{v}"].strip().lower(),
                                 f"rewrite_{v}", df, n_docs)
            errors.setdefault(qid, []).extend(e)
            warnings.setdefault(qid, []).extend(w)
        if row["rewrite_a"].strip() and tokens(row["rewrite_a"]) == tokens(row["rewrite_b"]):
            errors[qid].append("rewrite_a and rewrite_b are the same")
        if row["verified"].strip().lower() != "yes":
            warnings[qid].append("not verified yet (second pass)")
    return ({q: e for q, e in errors.items() if e}, {q: w for q, w in warnings.items() if w})


# ---------------------------------------------------------------- files

def sheet_path(cfg: dict) -> Path:
    return repo_dir(cfg, "queries_dir") / "variants" / f"{cfg['dataset']['name']}_rewrites.csv"


def read_sheet(path: Path) -> list[dict]:
    # utf-8-sig: Excel's "CSV UTF-8" adds a byte-order mark. Excel's plain "CSV"
    # option saves as Windows-1252 instead (curly quotes become byte 0x92 etc.).
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
        print(f"note: {path.name} is Windows-1252 encoded (Excel 'CSV'); read it as such")
    rows = [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(io.StringIO(text, newline=""))]
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return rows


def write_sheet(rows: list[dict], path: Path) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_topics_jsonl(cfg: dict) -> dict[str, dict]:
    path = repo_dir(cfg, "queries_dir") / f"{cfg['dataset']['name']}_topics.jsonl"
    with open(path, encoding="utf-8") as f:
        return {t["query_id"]: t for t in map(json.loads, f) if t}


def document_frequencies(cfg: dict) -> tuple[dict[str, int], int]:
    """Term -> number of documents containing it, over title + body. Cached in .cache/."""
    cache = REPO_ROOT / ".cache" / f"{cfg['dataset']['name']}_df.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data["df"], data["n_docs"]

    import ir_datasets
    from src.prepare_corpus import make_record

    ds = cfg["dataset"]
    print("Counting document frequencies over the corpus (once; cached afterwards)...")
    df: Counter = Counter()
    n_docs = 0
    for doc in ir_datasets.load(ds["docs_id"]).docs_iter():
        record = make_record(doc.doc_id, getattr(doc, ds["title_field"]), getattr(doc, ds["body_field"]))
        if record:
            df.update(set(tokens(record["contents"])))
            n_docs += 1
    ensure_parent(cache)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump({"n_docs": n_docs, "df": dict(df)}, f)
    return dict(df), n_docs


# ---------------------------------------------------------------- commands

def make_template(cfg: dict, force: bool = False) -> Path:
    path = sheet_path(cfg)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (it may contain your rewrites); not overwriting")
    df, n_docs = document_frequencies(cfg)
    topics = load_topics_jsonl(cfg)
    originals = read_queries(default_queries_path(cfg))
    rows = []
    for qid in sorted(originals, key=qid_sort_key):
        topic = topics.get(qid, {})
        rows.append({
            "qid": qid, "original": originals[qid],
            "question": topic.get("text", topic.get("description", "")),
            "narrative": topic.get("narrative", ""),
            "anchor": anchor_term(originals[qid], df, n_docs),
            **{c: "" for c in COLUMNS[5:]},
        })
    return write_sheet(rows, path)


def run_check(cfg: dict) -> bool:
    name = cfg["dataset"]["name"]
    rows = read_sheet(sheet_path(cfg))
    df, n_docs = document_frequencies(cfg)
    originals = read_queries(default_queries_path(cfg))
    for row in rows:  # always compare against the official original, not an edited copy
        if row["qid"] in originals:
            row["original"] = originals[row["qid"]]
    errors, warnings = check_rows(rows, set(originals), df, n_docs)

    for qid in sorted(set(errors) | set(warnings), key=qid_sort_key):
        original = originals.get(qid, "?")
        print(f"\n[{qid}] {original}")
        for e in errors.get(qid, []):
            print(f"  ERROR    {e}")
        for w in warnings.get(qid, []):
            print(f"  warning  {w}")

    done = sum(bool(r[f"rewrite_{v}"]) for r in rows for v in "ab")
    verified = sum(r["verified"].lower() == "yes" for r in rows)
    counts = Counter(r[f"phenomenon_{v}"].lower() for r in rows for v in "ab" if r[f"rewrite_{v}"])
    print(f"\n{done}/{2 * len(originals)} rewrites written, {verified}/{len(originals)} topics verified")
    print("phenomena:", dict(counts.most_common()))
    print(f"{sum(map(len, errors.values()))} errors, {sum(map(len, warnings.values()))} warnings")

    if errors:
        print("Fix the errors to export the rewrite query files.")
        return False

    queries_dir = repo_dir(cfg, "queries_dir")
    feature_rows = []
    for v in "ab":
        write_queries({r["qid"]: r[f"rewrite_{v}"] for r in rows}, queries_dir / f"{name}_rewrite_{v}.tsv")
        for r in rows:
            feature_rows.append({"qid": r["qid"], "variant": v, "original": r["original"],
                                 "rewrite": r[f"rewrite_{v}"], "phenomenon": r[f"phenomenon_{v}"].lower(),
                                 "verified": r["verified"].lower() == "yes",
                                 **features(r["original"], r[f"rewrite_{v}"], df, n_docs)})
    out = ensure_parent(queries_dir / "variants" / f"{name}_rewrite_features.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"Exported {name}_rewrite_a.tsv, {name}_rewrite_b.tsv and {out.name}")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_template = add_config_args(sub.add_parser("template", help="create the rewrite sheet"))
    p_template.add_argument("--force", action="store_true", help="overwrite an existing sheet (loses your rewrites!)")
    add_config_args(sub.add_parser("check", help="check the sheet and export query files"))
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if not cfg.get("dataset"):
        parser.error("--dataset is required")
    if args.command == "template":
        print(f"Wrote {make_template(cfg, args.force)}")
    else:
        run_check(cfg)


if __name__ == "__main__":
    main()
