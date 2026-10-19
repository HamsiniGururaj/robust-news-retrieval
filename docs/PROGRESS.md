# Project Progress and Decision Log

Group 5 · Information Retrieval · Mentor: Abhinav Kumar
Repository: https://github.com/HamsiniGururaj/robust-news-retrieval

This file records what has been built, what changed since the proposal (`docs/proposal.pdf`, submitted 21 Sep 2026), and why. It is the source for the intermediate status report and the final report.

---

## 1. Research goal (unchanged)

A controlled robustness study of **BM25**, a **dense bi-encoder (E5-base-v2)** and a **hybrid (reciprocal rank fusion)** when the *same* information need is worded differently. The document collection and the information need stay fixed; only the query wording changes.

| RQ | Question | Status of method |
|---|---|---|
| RQ1 | Baseline effectiveness on original topics | Unchanged |
| RQ2 | Effect of meaning-preserving, low-overlap rewrites | Unchanged |
| RQ3 | Which phenomena (paraphrase, acronym/alias, entity description, temporal expression, lexical de-anchoring) predict the lexical-vs-dense advantage | Unchanged |
| RQ4 | Negation sensitivity on NevIR, kept separate from benchmark scores | Unchanged |
| RQ5 | Whether RRF recovers relevant documents missed by either retriever | Unchanged |

Fixed system settings (preregistered, not tuned on test topics): BM25 k1=0.9, b=0.4, no stemming, stopwords kept; E5-base-v2 with `query: `/`passage: ` prefixes, max 512 tokens, L2-normalized, FAISS exact inner product; RRF k=60 over the top 1000 of each system. Metrics: nDCG@10 (primary), MAP, Recall@100, always per topic; paired bootstrap/randomization tests with 95% CIs and Holm correction.

---

## 2. Change of primary dataset: Core 2018 / WaPo v2 → TREC-COVID

### 2.1 What the proposal planned
TREC 2018 Common Core: 50 NIST topics with deep graded judgements over the **Washington Post Corpus v2** (~608k news articles). The proposal flagged this as its main risk: the corpus is controlled-access, and its mitigation was to build the pipeline on an open corpus while waiting and to ask the mentor whether to change the primary benchmark (proposal §10–11).

### 2.2 Why it is not available
The WaPo corpus is distributed only by NIST, after the requesting **organization** signs and emails an agreement (signed by the person responsible for the data at the institution) and each researcher signs an individual agreement. NIST asks requesters to allow **seven business days** for a reply; the data then arrives as a password-protected download. The corpus had not been obtained by 19 Oct 2026, which leaves too little time to index, encode, write rewrites and analyse before the mid-November submission. *(A last-minute request was prepared on 19 Oct; if access arrives, see §2.6.)*

### 2.3 What we looked for in a replacement
A replacement had to keep the study valid, so the requirements were:
1. **Real human relevance judgements**, deep enough for MAP and Recall@100 (the rewrite study reuses them, so they must be trustworthy).
2. **Freely downloadable today**, with no access request.
3. **TREC-style topics** with a short query *and* a narrative, so rewriters can verify that intent is preserved.
4. **Graded labels** (for nDCG) and roughly 50 topics, as in Core 2018.
5. **Small enough to index and encode on Colab** within our timeline.

### 2.4 Candidates considered

| Dataset | News? | Freely available? | Judgements | Decision |
|---|---|---|---|---|
| **TREC-COVID (BEIR)** | No (scientific abstracts) | Yes | 50 topics, 66,336 judgements, graded 0/1/2 | **Chosen** |
| CODEC | Web articles on economics/history/politics | No: access by emailing the authors | 42 topics, ~6k judgements (0–3) | Same access risk as WaPo, shallow judgements |
| NeuCLIR 2022/23 | Yes (CommonCrawl news) | Yes, but documents are Persian/Russian/Chinese | Deep | Needs machine-translated English (a confound); 2.2M+ docs too large for Colab |
| Signal-1M | Yes (1M articles, Sept 2015) | Yes (form) | None for query → article search (its benchmark retrieves *tweets*) | No usable qrels; kept for an optional small news set (§2.5) |
| Robust04, NYT (Core 2017), TREC News | Yes | No: licensed (TREC disks / LDC / WaPo) | Deep | Same licensing problem |
| TREC DL 2019/20 (MS MARCO) | No (web) | Yes | Deep, 43/54 topics | 8.8M passages: too large to encode on Colab |

No freely downloadable English news collection with ad-hoc relevance judgements was found: every standard English news test collection is licensed.

### 2.5 Why TREC-COVID
- **Same evaluation structure as Core 2018.** 50 NIST topics; each has a short keyword `query` (the analogue of a Core title), a `question` and a `narrative`. Judgements are deep (~1,300 per topic) and graded on the same 0/1/2 scale, so the metric definitions are unchanged.
- **Built for evaluation by NIST** (five rounds in 2020), widely used (it is part of BEIR), so results are comparable to published work.
- **Good fit for vocabulary mismatch.** Biomedical text is full of aliases and acronyms (COVID-19 / SARS-CoV-2 / coronavirus, ACE2 inhibitors, drug names vs. drug classes), so acronym/alias and entity-description rewrites are natural.
- **Feasible.** 171,332 documents (title + abstract) can be indexed and encoded on a single Colab GPU.

**What changes as a result**
- The domain is scientific, not news, so the working title becomes *Robust Ad-hoc Retrieval Under Vocabulary Mismatch: Lexical, Dense and Hybrid Retrieval* (to be confirmed with the mentor). The method, systems, metrics and RQs are unchanged.
- Documents are short (title + abstract), so the proposal's 512-token truncation concern largely disappears. We report this as a difference from the original design.
- Temporal-expression rewrites will be rarer than in news; the other four phenomena are well represented.
- **Pooling bias.** The BEIR authors note that TREC-COVID's judgement pools came mostly from lexical systems, so dense retrievers return more *unjudged* documents, which count as non-relevant. We therefore also report **Judged@10** (the share of each top 10 that was judged) for every run, so lower dense scores can be checked against missing judgements before being interpreted.
- Two judgements in the BEIR qrels carry the label −1; they are stored as 0 (not relevant).

**Optional news component.** Signal-1M (1M real news articles, already downloaded, kept in Drive only) can host the proposal's "small independently judged news exclusion set" (pipeline stage 8) for RQ4, if time allows.

### 2.6 If WaPo access arrives late
All code takes the dataset from a config file (`configs/core18.yaml` already exists and the Core 2018 topics and qrels are exported), so Core 2018 can be run as a second benchmark without code changes. Its 608k documents would need several Colab sessions for dense encoding, which the sharded encoder is designed for.

---

## 3. What has been built

### 3.1 Engineering setup
| Component | Purpose |
|---|---|
| GitHub repo (private) | Single source of code; data never committed |
| `notebooks/00_setup.ipynb` | Colab setup: mount Drive, clone repo with a token from Colab Secrets, install Java 21 + requirements |
| `requirements.txt`, `requirements-lock.txt` | Dependencies; exact versions from the working Colab environment |
| `docs/WORKFLOW.md` | How the team writes (VS Code), stores (GitHub) and runs (Colab + Drive) code |
| `docs/ROADMAP.md` | Step-by-step implementation plan |
| `pyproject.toml`, `tests/` | Local test suite (pytest); 69 tests passing |

### 3.2 Code modules

| Step | Module | What it does | Status |
|---|---|---|---|
| A0 | `configs/*.yaml`, `src/config.py` | All settings in one place; per-dataset configs; command-line overrides that reject unknown keys | Done, tested |
| A0 | `src/io_utils.py` | One reader/writer for queries (TSV), qrels and runs (TREC format), and JSON manifests recording config, git commit and timings for every output; score ties ordered exactly as trec_eval orders them | Done, tested |
| A1 | `src/load_topics.py` | Exports topics and qrels from `ir_datasets` into `queries/` (one file per query field, a full-topic JSONL with narratives for rewriters, qrels), with a data health check | Done, tested; run on Core 2018, SciFact, TREC-COVID |
| A2 | `src/evaluate.py` | nDCG@10, MAP, Recall@100 and Judged@10 per topic and mean via `ir_measures` (pytrec_eval backend); topics missing from a run score 0 | Done, tested against hand-computed scores |
| A3 | `src/prepare_corpus.py` | Documents → one JSONL (`id`, `title`, `body`, `contents`) shared by BM25 and dense; skips empty docs and repeated ids; atomic write; length statistics | Done, tested; run locally on TREC-COVID (171,331 docs, 13.5 s) |
| A4 | `src/bm25.py` | Pyserini/Lucene index over title + body with no stemming and stopwords kept (verified automatically after indexing); BM25 k1=0.9, b=0.4 with matching query analysis; top 1000; per-query latency, index size and build time recorded | Done; run in Colab on SciFact and TREC-COVID (results in §3.5) |
| A5 | `src/dense_encode.py`, `src/dense_search.py` | E5-base-v2 encoding of `passage: ` + title + body (512 tokens, fp16 on GPU, L2-normalized) in 50k-document shards saved to Drive and resumable after a disconnect; exact FAISS IndexFlatIP search with `query: ` prefix; latency, memory, encoding time and model commit recorded | Code done, tested locally with a fake encoder; first Colab run pending |
| A6 | `src/fuse.py` | Reciprocal rank fusion (k=60, top 1000 of each run) | Code done, tested against hand-computed scores; first Colab run pending |

### 3.3 Corpus facts (TREC-COVID, from `prepare_corpus`)
- 171,331 documents kept (1 empty document dropped); 42,139 (25%) are title-only.
- Median 168 words per document; only 1.75% exceed 400 words. E5's 512-token input limit therefore almost never truncates, unlike the long WaPo articles the proposal planned for.

### 3.4 Exported data
| Dataset | Topics | Judgements | Labels | Role |
|---|---|---|---|---|
| TREC-COVID | 50 | 66,336 | 0/1/2 | Primary benchmark |
| SciFact | 300 | 339 | 1 only | Pipeline smoke test |
| Core 2018 | 50 | 26,233 | 0/1/2 | Exported; used only if WaPo arrives |

---

### 3.5 Preliminary results: BM25 (Colab, 19 Oct 2026)

| Dataset | Queries | nDCG@10 | MAP | Recall@100 | Judged@10 |
|---|---|---|---|---|---|
| SciFact (smoke test) | claims | 0.665 | 0.627 | 0.885 | 0.086* |
| TREC-COVID | keyword `query` (original) | **0.639** | 0.203 | 0.114 | 0.958 |
| TREC-COVID | question `text` | 0.584 | 0.155 | 0.097 | 0.932 |

\* SciFact qrels list only relevant documents, so Judged@10 is not meaningful there.

- **Sanity checks pass.** SciFact matches published BM25 (~0.665–0.68). On TREC-COVID, BEIR's standard setting uses the question form, where published flat BM25 is ~0.59; ours is 0.584 (no stemming, by design).
- **Keyword queries beat questions by 0.055 nDCG@10** for BM25: question words dilute the matching. A first, informal sign of how much BM25 depends on wording.
- **Low MAP and Recall@100 are expected:** topics have a median of 478 relevant documents, so the top 100 can only hold a fraction of them. nDCG@10 is the primary metric.
- **Index:** the self-check confirmed no stemming and stopwords kept. 321 MB, 244k unique terms, built in 45 s. Latency (~115 ms/query in this first run) was inflated by reading the index over the Drive mount; it is now measured on a local copy after a warm-up query.

## 4. Pending work
See `docs/ROADMAP.md` for the full plan.

| Step | Component | Notes |
|---|---|---|
| A7 | End-to-end runs | SciFact, then TREC-COVID; efficiency (index size, encoding time, latency) recorded in manifests |
| A8 | Significance tests (`stats.py`) | Paired bootstrap / randomization, Holm |
| A9 | NevIR (RQ4) | BM25, E5, cross-encoder reference |
| A10 | Error-analysis tool | Top-10 inspection with labels |
| A11 | Rewrite tooling | Validate the rewrite CSV, compute overlap features |
| — | Rewrites | 2 meaning-preserving rewrites × 50 TREC-COVID topics, tagged and cross-checked |
| B | Analysis | RQ2/RQ3 robustness, RQ5 complementarity, figures, report |
