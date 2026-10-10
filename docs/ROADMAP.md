# Code Roadmap

How we build the code for *Robust News Retrieval Under Vocabulary Mismatch*, step by step. See `docs/proposal.pdf` for the research design and `docs/WORKFLOW.md` for how we use VS Code, GitHub, Colab and Drive.

## Principles

- **Small, testable steps.** Each step is one or two modules, with a test and a commit, checked in Colab before we move on.
- **Shared file formats.** Every retriever writes TREC run files. Evaluation, fusion, statistics and analysis read only run files and qrels, so they work the same for every system.
- **SciFact first, WaPo later.** Every script takes the dataset from a config. Switching to WaPo means changing the config, not the code.
- **No tuning on Core 2018.** BM25 (k1=0.9, b=0.4), RRF (k=60) and the E5 settings stay fixed as preregistered. SciFact is only for checking that the pipeline works.
- **WaPo-independent work first.** We don't have the WaPo corpus yet, so Track A covers everything that doesn't need it.

## Decisions

| Decision | Choice |
|---|---|
| Original query | Topic **title**. Description runs are secondary only. Rewrites are of the title. |
| Local vs Colab | Lightweight modules (I/O, evaluation, fusion, statistics) are tested locally with `pytest`. Pyserini, GPU work and large data run in Colab. |
| BM25 on NevIR | Each NevIR item has only two documents, so BM25 statistics come from the pool of all NevIR documents. This is stated in the report. |
| Scope | The cross-encoder reference on NevIR is in scope. The hand-judged news exclusion set is stretch. |

## Shared formats

| Thing | Format | Location |
|---|---|---|
| Queries | TSV: `qid \t text` | `queries/` |
| Query variant metadata | CSV: `qid, variant_id, text, phenomenon, author, verifier, notes` | `queries/variants/` |
| Qrels | Standard TREC qrels | `queries/` (Core 2018 qrels are public) |
| Runs | TREC 6-column; `run_tag` = system + query set | `runs/<dataset>/` |
| Per-topic metrics | Long CSV: `run, qid, measure, value` | `results/` |
| Corpus | JSONL `{id, title, body, contents}` | Google Drive only |
| Run manifest | JSON next to each run: config, git commit, model revision, timings | `runs/` |

**Rewrite runs keep the original qids.** The rewrite of topic 321 is still qid `321` in the run for rewrite set A. The same qrels then apply unchanged, and pairing a rewrite with its original is a join on qid. Phenomenon tags live in the variant CSV.

---

## Track A: before WaPo (now)

| # | Step | Files | Done when |
|---|---|---|---|
| A0 | **Foundations**: configs, path handling, shared I/O, test setup | `configs/*.yaml`, `src/config.py`, `src/io_utils.py`, `tests/` | I/O round-trip tests pass locally |
| A1 | **Topics and qrels**: Core 2018 (public, no corpus needed) and SciFact | `src/load_topics.py` | `queries/core18_title.tsv` etc. written; 50 topics, each with relevant documents |
| A2 | **Evaluation**: nDCG@10, MAP and Recall@100, per topic and mean; topics missing from a run score 0 | `src/evaluate.py` | Test on a made-up run matches scores worked out by hand |
| A3 | **Corpus prep**: documents → JSONL (id, title, body, contents) | `src/prepare_corpus.py` | SciFact JSONL in Drive |
| A4 | **BM25**: Pyserini index and search; no stemming or stopword removal at index *and* query time; k1=0.9, b=0.4; top 1000 | `src/bm25.py` | SciFact BM25 run evaluated |
| A5 | **Dense**: E5-base-v2 (`passage: ` / `query: ` prefixes, max length 512, L2-normalized), sharded and resumable encoding saved to Drive, FAISS `IndexFlatIP` search | `src/dense_encode.py`, `src/dense_search.py` | SciFact dense run evaluated; an interrupted encode resumes |
| A6 | **RRF**: k=60 over the top 1000 of each run; ties broken by doc id | `src/fuse.py` | Fusion test passes; hybrid run evaluated |
| A7 | **SciFact smoke test**: the full pipeline in one notebook | `notebooks/01_smoke_scifact.ipynb` | Scores land near published BEIR SciFact figures; RRF ≥ best single system |
| A8 | **Statistics**: paired bootstrap, paired randomization test, 95% CIs, Holm correction, effect sizes | `src/stats.py` | Tests on synthetic data with known answers pass |
| A9 | **NevIR (RQ4)**: BM25, E5 and a cross-encoder; pairwise accuracy; results kept separate from TREC | `src/nevir.py`, `notebooks/03_nevir.ipynb` | RQ4 results finished |
| A10 | **Error-analysis tool**: top-10 results with labels, plus relevant docs each system missed | `src/inspect_runs.py` | Works on SciFact runs |
| A11 | **Variant tooling**: validate the rewrite CSV; compute overlap, length, number and entity features | `src/variants.py` | Validates the team's draft rewrites |

### Team work in parallel (no code needed)

Starting once A1 has produced `queries/core18_title.tsv`:

- Write **two meaning-preserving rewrites** for each of the 50 Core 2018 titles (proposal §3.1).
- Tag each rewrite with one phenomenon: paraphrase, acronym/alias, entity description, temporal expression, or lexical de-anchoring.
- A **different** teammate checks each rewrite against the rubric.
- **Not allowed:** negation, number swaps and entity substitutions, because they change relevance.
- The CSV template comes with A11.

---

## Track B: after WaPo arrives

| # | Step |
|---|---|
| B1 | Link the WaPo tarball from Drive; prepare the corpus (handling duplicate or empty documents); build the BM25 index |
| B2 | Dense encoding on a GPU runtime in shards, over several sessions; log index size and encoding time |
| B3 | **RQ1**: BM25, dense and RRF on the 50 titles, plus evaluation and significance tests |
| B4 | IDF-based variant features (anchor-term check, max IDF) from the WaPo index; finalize the rewrites |
| B5 | Runs on rewrite sets A and B |
| B6 | **RQ2/RQ3**: change in nDCG@10 from original to rewrite, CIs, worst-case drops, breakdown by phenomenon and overlap bin, feature regression, ranking stability |
| B7 | **RQ5**: complementarity (relevant docs found by only one system, oracle union recall, RRF gain on worst topics) |
| B8 | Error analysis, figures, report |

## Track C: stretch, if time allows

- Cross-encoder reranking of the hybrid top 50
- Passage-level dense retrieval with max-passage scoring (truncation ablation)
- Query-adaptive fusion, trained on a development split only
- Hand-judged news exclusion set (pooling, blinding, kappa)
- TREC News background linking (transfer evaluation)
