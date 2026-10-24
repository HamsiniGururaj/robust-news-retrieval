# Code Roadmap

How we build the code for *Robust Retrieval Under Vocabulary Mismatch*, step by step. See `docs/proposal.pdf` for the original design, `docs/PROGRESS.md` for what is done and why the dataset changed, and `docs/WORKFLOW.md` for how we use VS Code, GitHub, Colab and Drive.

## Key dates

| Date | Milestone |
|---|---|
| 25 Oct 2026 (Sun) | Intermediate status report (4-page PDF): needs BM25 / dense / RRF results on TREC-COVID |
| Mid-Nov 2026 | Final submission |
| Week after | 10–15 min presentation and demo |

## Principles

- **Small, testable steps.** Each step is one or two modules, with a test and a commit, checked in Colab before we move on.
- **Shared file formats.** Every retriever writes TREC run files. Evaluation, fusion, statistics and analysis read only run files and qrels, so they work the same for every system.
- **SciFact first, then TREC-COVID.** Every script takes the dataset from a config (`--dataset scifact|trec_covid|core18`). Switching datasets means changing the config, not the code.
- **No tuning on test topics.** BM25 (k1=0.9, b=0.4), RRF (k=60) and the E5 settings stay fixed as preregistered. SciFact only checks that the pipeline works.

## Decisions

| Decision | Choice |
|---|---|
| Primary benchmark | **TREC-COVID (BEIR)**, replacing Core 2018 / WaPo v2 (see `docs/PROGRESS.md` §2) |
| Original query | The topic's short keyword **`query`** field (analogue of a Core title). The question form (`text`) is secondary. Rewrites are of the keyword query. |
| Unjudged documents | Every run also reports **Judged@10**, because TREC-COVID pools favour lexical systems |
| Local vs Colab | Lightweight modules (I/O, evaluation, fusion, statistics) are tested locally with `pytest`. Pyserini, GPU work and large data run in Colab. |
| BM25 on NevIR | Each NevIR item has only two documents, so BM25 statistics come from the pool of all NevIR documents. This is stated in the report. |
| Scope | The cross-encoder reference on NevIR is in scope. A small hand-judged news exclusion set on Signal-1M is stretch. |
| Git | One person codes for now, so commits go straight to `main` |

## Shared formats

| Thing | Format | Location |
|---|---|---|
| Queries | TSV: `qid \t text` | `queries/` |
| Query variant metadata | CSV: `qid, variant_id, text, phenomenon, author, verifier, notes` | `queries/variants/` |
| Qrels | Standard TREC qrels | `queries/` |
| Runs | TREC 6-column; `run_tag` = system + query set | `runs/<dataset>/` |
| Per-topic metrics | Long CSV: `run, qid, measure, value` | `results/<dataset>/` |
| Corpus | JSONL `{id, title, body, contents}` | Google Drive only |
| Run manifest | JSON next to each run: config, git commit, model revision, timings | `runs/` |

**Rewrite runs keep the original qids.** The rewrite of topic 7 is still qid `7` in the run for rewrite set A. The same qrels then apply unchanged, and pairing a rewrite with its original is a join on qid. Phenomenon tags live in the variant CSV.

---

## Track A: pipeline (now → 25 Oct)

| # | Step | Files | Done when | Status |
|---|---|---|---|---|
| A0 | **Foundations**: configs, path handling, shared I/O, test setup | `configs/*.yaml`, `src/config.py`, `src/io_utils.py`, `tests/` | I/O round-trip tests pass locally | ✅ |
| A1 | **Topics and qrels** for TREC-COVID, SciFact (and Core 2018) | `src/load_topics.py` | `queries/trec_covid_query.tsv` etc. written | ✅ |
| A2 | **Evaluation**: nDCG@10, MAP, Recall@100, Judged@10 per topic and mean; topics missing from a run score 0 | `src/evaluate.py` | Test on a made-up run matches scores worked out by hand | ✅ |
| A3 | **Corpus prep**: documents → JSONL (id, title, body, contents) | `src/prepare_corpus.py` | SciFact and TREC-COVID JSONL in Drive | ✅ code; tested on TREC-COVID locally |
| A4 | **BM25**: Pyserini index and search; no stemming or stopword removal at index *and* query time; k1=0.9, b=0.4; top 1000 | `src/bm25.py` | BM25 runs evaluated | ✅ SciFact 0.665, TREC-COVID 0.639 nDCG@10 |
| A5 | **Dense**: E5-base-v2 (`passage: ` / `query: ` prefixes, max length 512, L2-normalized), sharded and resumable encoding saved to Drive, FAISS `IndexFlatIP` search | `src/dense_encode.py`, `src/dense_search.py` | Dense runs evaluated; an interrupted encode resumes | ✅ SciFact 0.721, TREC-COVID 0.563 / 0.697 |
| A6 | **RRF**: k=60 over the top 1000 of each run; ties broken by doc id | `src/fuse.py` | Fusion test passes; hybrid run evaluated | ✅ TREC-COVID 0.731 / 0.757 |
| A7 | **End-to-end runs**: SciFact smoke test, then TREC-COVID (RQ1); index size, encoding time, query latency recorded | `notebooks/01_pipeline.ipynb` | SciFact near published BEIR figures; TREC-COVID RQ1 table | ✅ 19 Oct (PROGRESS §3.5) |
| A8 | **Statistics**: paired bootstrap, paired randomization test, 95% CIs, Holm correction, effect sizes | `src/stats.py` | Tests on synthetic data with known answers pass | ✅ run on TREC-COVID (PROGRESS §3.5) |

### Schedule to the intermediate report

| Day | Work | Status |
|---|---|---|
| 19 Oct | A3–A8: corpus, BM25, dense, RRF, end-to-end runs, significance tests | ✅ (ahead of plan) |
| 20–21 Oct | A11 tooling + 100 query rewrites, verified | ✅ |
| 21 Oct | Rewrite runs (notebook Part 4); RQ2 significance on 3 measures into PROGRESS §3.7 | ✅ (ahead of plan) |
| 23 Oct | Draft the 4-page report from `docs/PROGRESS.md` and the results | |
| 24 Oct | Finalize the PDF; contribution matrix | |
| 25 Oct | Submit (`5_<ShortTitle>.pdf`) | |

### Query rewrites ✅ (21 Oct)

100 meaning-preserving rewrites (2 × 50 topics) written, tagged with one phenomenon each, and self-verified against the narratives (single annotator; see `docs/REWRITE_GUIDE.md` and PROGRESS §3.6). Sheet: `queries/variants/trec_covid_rewrites.csv`; check with `python -m src.variants check --dataset trec_covid`.

---

## Track B: after the intermediate report (→ mid-Nov)

| # | Step |
|---|---|
| A9 | **NevIR (RQ4)**: BM25, E5 and a cross-encoder; pairwise accuracy; reported separately |
| A10 | **Error-analysis tool**: top-10 results with labels, plus relevant docs each system missed |
| A11 | **Variant tooling**: validate the rewrite CSV; overlap, IDF, length, number and entity features. ✅ done; 100 rewrites finished 21 Oct |
| B1 | Runs on rewrite sets A and B ✅ 21 Oct; first RQ2 results in PROGRESS §3.7 |
| B2 | **RQ2/RQ3**: ✅ first results (PROGRESS §3.7, §3.8; `src/robustness.py`). Remaining: per-topic failure examples, ranking stability |
| B3 | **RQ5**: complementarity (relevant docs found by only one system, oracle union recall, RRF gain on worst topics) |
| B4 | Error analysis, figures, final report, presentation and demo |

## Track C: stretch, if time allows

- Cross-encoder reranking of the hybrid top 50
- Small hand-judged news exclusion set on Signal-1M (pooling, blinding, kappa) for RQ4
- Core 2018 / WaPo as a second benchmark, if NIST access arrives
- Query-adaptive fusion, trained on a development split only
- Passage-level dense retrieval (less relevant now: TREC-COVID documents are short)
