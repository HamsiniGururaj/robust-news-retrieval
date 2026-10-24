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
| `docs/REWRITE_GUIDE.md` | Rubric and procedure for writing and verifying the query rewrites |
| `notebooks/01_pipeline.ipynb` | Colab driver: setup → SciFact smoke test → TREC-COVID BM25 → dense + RRF → evaluation → efficiency → rewrite runs (Part 4) |
| `pyproject.toml`, `tests/` | Local test suite (pytest); 100 tests passing |

### 3.2 Code modules

| Step | Module | What it does | Status |
|---|---|---|---|
| A0 | `configs/*.yaml`, `src/config.py` | All settings in one place; per-dataset configs; command-line overrides that reject unknown keys | Done, tested |
| A0 | `src/io_utils.py` | One reader/writer for queries (TSV), qrels and runs (TREC format), and JSON manifests recording config, git commit and timings for every output; score ties ordered exactly as trec_eval orders them | Done, tested |
| A1 | `src/load_topics.py` | Exports topics and qrels from `ir_datasets` into `queries/` (one file per query field, a full-topic JSONL with narratives for rewriters, qrels), with a data health check | Done, tested; run on Core 2018, SciFact, TREC-COVID |
| A2 | `src/evaluate.py` | nDCG@10, MAP, Recall@100 and Judged@10 per topic and mean via `ir_measures` (pytrec_eval backend); topics missing from a run score 0 | Done, tested against hand-computed scores |
| A3 | `src/prepare_corpus.py` | Documents → one JSONL (`id`, `title`, `body`, `contents`) shared by BM25 and dense; skips empty docs and repeated ids; atomic write; length statistics | Done, tested; run locally on TREC-COVID (171,331 docs, 13.5 s) |
| A4 | `src/bm25.py` | Pyserini/Lucene index over title + body with no stemming and stopwords kept (verified automatically after indexing); BM25 k1=0.9, b=0.4 with matching query analysis; top 1000; per-query latency, index size and build time recorded | Done; run in Colab on SciFact and TREC-COVID (results in §3.5) |
| A5 | `src/dense_encode.py`, `src/dense_search.py` | E5-base-v2 encoding of `passage: ` + title + body (512 tokens, fp16 on GPU, L2-normalized) in 50k-document shards saved to Drive and resumable after a disconnect; exact FAISS IndexFlatIP search with `query: ` prefix; latency, memory, encoding time and model commit recorded | Done; run in Colab on SciFact and TREC-COVID (results in §3.5) |
| A6 | `src/fuse.py` | Reciprocal rank fusion (k=60, top 1000 of each run) | Done; run in Colab (results in §3.5) |
| A8 | `src/stats.py` | Paired randomization test (p-values), paired bootstrap (95% CI of the mean difference), Holm correction across comparisons, wins/ties/losses per topic | Done, tested on synthetic data; run on TREC-COVID (§3.5) |
| RQ3 | `src/robustness.py` | Change per system per rewrite; dense advantage by phenomenon (bootstrap CI, randomization test, Holm) and its rank correlation with rewrite features (permutation test, Holm) | Done, tested; results in §3.8 |
| A11 | `src/variants.py`, `docs/REWRITE_GUIDE.md` | Rewrite sheet pre-filled with each topic's original query, question, narrative and anchor (rarest) term; automatic checks (negation added, phenomenon tag, anchor still present, content-word overlap, length and question/keyword form, standalone numbers, verification); exports the rewrite query files and a per-rewrite feature table (overlap, anchor removed, IDF, length, numbers, acronyms) for RQ3 | Done, tested; used to produce the final rewrite set (§3.6). Also reads sheets saved by Excel as Windows-1252 |

### 3.3 Corpus facts (TREC-COVID, from `prepare_corpus`)
- 171,331 documents kept (1 empty document dropped); 42,139 (25%) are title-only.
- Median 168 words per document; only 1.75% exceed 400 words. E5's 512-token input limit therefore almost never truncates, unlike the long WaPo articles the proposal planned for.

### 3.4 Exported data
| Dataset | Topics | Judgements | Labels | Role |
|---|---|---|---|---|
| TREC-COVID | 50 | 66,336 | 0/1/2 | Primary benchmark |
| SciFact | 300 | 339 | 1 only | Pipeline smoke test |
| Core 2018 | 50 | 26,233 | 0/1/2 | Exported; used only if WaPo arrives |

Query sets for TREC-COVID: `trec_covid_query.tsv` (original keyword queries), `trec_covid_text.tsv` (NIST question form), `trec_covid_rewrite_a.tsv` and `trec_covid_rewrite_b.tsv` (our rewrites, §3.6).

---

### 3.5 Preliminary results (Colab, 19 Oct 2026)

Fixed, untuned settings throughout (BM25 k1=0.9, b=0.4; E5-base-v2; RRF k=60). Significance tests (A8) follow the results table.

**SciFact (smoke test, 300 queries):** BM25 0.665, dense 0.721, RRF 0.722 nDCG@10. Both single systems match published figures (BM25 ~0.665–0.68; E5-base-v2 ~0.72), so the pipeline produces correct numbers.

**TREC-COVID (50 topics):**

| System | Query form | nDCG@10 | MAP | Recall@100 | Judged@10 |
|---|---|---|---|---|---|
| BM25 | keyword (original) | 0.639 | 0.203 | 0.114 | 0.958 |
| BM25 | question | 0.584 | 0.155 | 0.097 | 0.932 |
| Dense (E5) | keyword (original) | 0.563 | 0.204 | 0.114 | 0.686 |
| Dense (E5) | question | 0.697 | 0.235 | 0.129 | 0.822 |
| **RRF** | keyword (original) | **0.731** | **0.262** | 0.133 | 0.936 |
| **RRF** | question | **0.757** | 0.257 | **0.137** | 0.978 |

Observations:
1. **RRF is the best system in both settings**, +0.09 (keyword) and +0.06 (question) nDCG@10 over the best single system. This is early support for H3/RQ5: BM25 and dense retrieve complementary relevant documents.
2. **Wording sensitivity, a first look at RQ2.** NIST supplies two wordings of every topic. Moving from keyword to question form changes nDCG@10 by −0.055 for BM25, **+0.134 for dense** and +0.025 for RRF. Dense is the *most* form-sensitive here, plausibly because E5 is trained mostly on natural-language questions, so bare keyword queries are out of distribution. RRF is the most stable. Note that the question form also adds words, unlike our controlled low-overlap rewrites.
3. **Pooling bias is visible.** 31% of dense's top 10 for keyword queries was never judged (Judged@10 0.686, vs 0.958 for BM25), so dense scores, especially with keyword queries, are underestimated by an unknown amount. All dense results are reported together with Judged@10.
4. **Low MAP and Recall@100 are expected:** topics have a median of 478 relevant documents, so the top 100 can hold only a fraction of them. nDCG@10 is the primary metric.

**Significance (TREC-COVID, nDCG@10, 50 topics).** Paired randomization test (10,000 resamples) for p, paired bootstrap for the 95% CI of the mean difference, Holm correction within each family of three comparisons. W/T/L = topics won/tied/lost by A.

| Family | A vs B | Δ nDCG@10 | 95% CI | p (Holm) | W/T/L |
|---|---|---|---|---|---|
| Systems, keyword queries | RRF vs BM25 | **+0.092** | [+0.046, +0.142] | **0.0008** | 33/4/13 |
| | RRF vs dense | **+0.169** | [+0.108, +0.232] | **0.0003** | 37/3/10 |
| | BM25 vs dense | +0.076 | [+0.001, +0.152] | 0.055 (n.s.) | 27/2/21 |
| Systems, question queries | RRF vs BM25 | **+0.173** | [+0.119, +0.229] | **0.0003** | 36/3/11 |
| | RRF vs dense | **+0.060** | [+0.012, +0.110] | **0.026** | 33/2/15 |
| | dense vs BM25 | **+0.113** | [+0.047, +0.180] | **0.0024** | 31/1/18 |
| Query form (question − keyword) | BM25 | −0.056 | [−0.131, +0.021] | 0.33 (n.s.) | 19/0/31 |
| | dense | **+0.134** | [+0.069, +0.201] | **0.0006** | 33/3/14 |
| | RRF | +0.025 | [−0.021, +0.073] | 0.33 (n.s.) | 28/1/21 |

- **RRF significantly beats both single systems** with both query forms, winning on 33–37 of 50 topics. This is the strongest preliminary result (H3).
- **BM25 vs dense depends on the query form.** With keyword queries BM25 is ahead, but not significantly; with question queries dense is significantly ahead.
- **Only dense is significantly sensitive to the query form.** BM25's and RRF's changes are within noise. Fusion absorbs the dense model's sensitivity.
- On MAP (keyword queries), BM25 and dense are indistinguishable (−0.001, p=0.94), and RRF beats both (p_holm=0.0003).

Commands: `python -m src.stats --dataset trec_covid --runs bm25_query dense_query rrf_query` (and `_text`); `--label query_form --pairs bm25_text:bm25_query dense_text:dense_query rrf_text:rrf_query`. Outputs: `results/trec_covid/significance_*.csv`.

**Efficiency (TREC-COVID, 171,331 documents; Colab, Tesla T4 GPU):**

| | BM25 (Lucene) | Dense (E5 + FAISS IndexFlatIP) | RRF |
|---|---|---|---|
| Build | 45 s indexing (CPU, 8 threads) | 814 s (13.6 min) encoding on GPU, fp16 | – |
| Index size | 321 MB (244k unique terms) | 502 MB (171,331 × 768 float32) | – |
| Load index | – | 1.5 s (exact FAISS index in RAM) | – |
| Query latency, mean (p95) | 93 ms (157 ms), CPU, top 1000 | 107 ms (121 ms): GPU query encoding + CPU exact search | +3 ms on top of both |

Latency is measured one query at a time after a warm-up query, on a local copy of the index. Dense latency is dominated by brute-force comparison against all 171k vectors (13 ms/query on SciFact's 5k documents).

Model revision used: `intfloat/e5-base-v2` @ `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` (now pinned in `configs/default.yaml`).

### 3.6 Query rewrite set (completed 21 Oct 2026)

100 rewrites, 2 per topic, of the original keyword queries, following `docs/REWRITE_GUIDE.md`. Written by a single annotator, checked automatically, then re-verified against each topic's narrative in a separate pass with the original hidden. A manual review of all 100 against their originals found 8 likely meaning changes (broadening, narrowing, a form change); these were corrected before any system was run on the rewrites.

| Property | Value |
|---|---|
| Rewrites | 100 (50 topics × A, B); all verified |
| Phenomena | paraphrase 42, de-anchoring 34, alias 16, entity description 7, temporal 1 |
| Set A / set B | A mostly paraphrase (35) and alias (14); B mostly de-anchoring (33), plus all 7 entity descriptions |
| Anchor (rarest original term) removed | 97 / 100 (the 3 kept are justified in the sheet's notes) |
| Content-word overlap with original (Jaccard) | mean 0.13, median 0.10; 49 rewrites share no content word |
| Length | 4.8 words on average vs 3.2 for the originals (mean change +1.5, range −2 to +4); keyword form kept throughout |
| Checker | 0 errors; 9 accepted warnings (length +4 ×3, anchor kept ×3, overlap > 0.5 ×3) |

Implications for the analysis:
- **Sets A and B are not interchangeable.** B is systematically more de-anchored, so results are reported per set as well as pooled.
- **Phenomenon groups for RQ3:** paraphrase, de-anchoring and alias are large enough to compare. Entity description (7) and temporal (1) are reported qualitatively, as the proposal's risk table anticipated.
- **Limitation:** single-annotator verification. The proposal planned independent verification by a second group member.

Files: `queries/variants/trec_covid_rewrites.csv` (the sheet), `queries/trec_covid_rewrite_{a,b}.tsv` (run inputs), `queries/variants/trec_covid_rewrite_features.csv` (per-rewrite features for RQ3).

### 3.7 RQ2 first results: original vs rewritten queries (21 Oct 2026)

Each system was run on rewrite sets A and B (`notebooks/01_pipeline.ipynb`, Part 4; GPU query encoding throughout) and scored against the unchanged qrels.

**Judged documents collapse for the rewrites.** The rewrites retrieve many documents that the 2020 NIST pools (built from systems run on the *original* topics) never judged, and the standard metrics count those as non-relevant:

| System | Judged@10: original | rewrite A | rewrite B |
|---|---|---|---|
| BM25 | 0.958 | 0.408 | 0.396 |
| Dense | 0.686 | 0.534 | 0.506 |
| RRF | 0.936 | 0.564 | 0.554 |

So every result is reported on three measures: standard **nDCG@10**, **judged-only nDCG'@10** (unjudged documents removed before scoring, "condensed lists", Sakai 2007) and **bpref** (built for incomplete judgements, Buckley & Voorhees 2004).

**Scores:**

| Run | nDCG@10 | nDCG'@10 (judged only) | Bpref |
|---|---|---|---|
| BM25 original / A / B | 0.639 / 0.303 / 0.240 | 0.653 / 0.536 / 0.530 | 0.329 / 0.163 / 0.172 |
| Dense original / A / B | 0.563 / 0.430 / 0.397 | 0.778 / 0.756 / 0.730 | 0.394 / 0.351 / 0.333 |
| RRF original / A / B | 0.731 / 0.454 / 0.395 | 0.759 / 0.715 / 0.680 | 0.407 / 0.312 / 0.297 |

**Change from original to rewrite (A / B), with Holm-corrected significance within each family of three systems:**

| System | nDCG@10 | nDCG'@10 | Bpref |
|---|---|---|---|
| BM25 | **−0.336 / −0.399** | **−0.117 / −0.123** | **−0.166 / −0.158** |
| Dense | **−0.133 / −0.166** | −0.022 / −0.047 (n.s.) | **−0.043 / −0.062** |
| RRF | **−0.277 / −0.337** | −0.044 (n.s.) / **−0.079** | **−0.095 / −0.110** |

Bold = significant at α = 0.05 after Holm.

**H1: does dense lose less than BM25?** Paired test on the per-topic changes, Holm across 6 change-vs-change comparisons per measure:

| Comparison (difference in drop) | nDCG@10 | nDCG'@10 | Bpref |
|---|---|---|---|
| Dense vs BM25, set A | **+0.203** (p_holm 0.0006, 34/16) | +0.095 (raw p 0.025, p_holm 0.15) | **+0.123** (p_holm 0.0006, 38/12) |
| Dense vs BM25, set B | **+0.234** (p_holm 0.001, 37/13) | +0.076 (raw p 0.098, p_holm 0.39) | **+0.096** (p_holm 0.0006, 38/12) |
| RRF vs BM25, set A / B | +0.059 / +0.062 (n.s.) | +0.073 / +0.044 (n.s.) | **+0.071 / +0.047** |
| RRF vs dense, set A / B | **−0.144 / −0.171** | −0.022 / −0.032 (n.s.) | **−0.052 / −0.049** |

Findings:
1. **H1 supported.** Dense loses much less than BM25 under meaning-preserving rewording on every measure (it wins on 33–38 of 50 topics). The difference is significant on nDCG@10 and bpref. On judged-only nDCG' it points the same way but does not survive Holm correction across six comparisons.
2. **BM25 is the system that rewording really hurts.** Its drop is significant on all three measures. Dense's judged-only drop (−0.02 / −0.05) is not significant: when only judged documents count, dense is close to wording-invariant.
3. **RRF is less robust than dense (nuance for H3).** RRF drops significantly more than dense on nDCG@10 and bpref, because fusion imports BM25's failures on the rewrites. On the rewrites RRF ≈ dense (nDCG@10 0.454 vs 0.430; 0.395 vs 0.397), so its large advantage on the original queries disappears. Fusion helps when both components work; it does not protect against one component collapsing. RQ5 will look at this per topic.
4. **Set B (mostly de-anchoring) hurts more than set A (mostly paraphrase/alias)** for every system, as expected.
5. **The judged-only metric reverses the RQ1 keyword-query comparison:** dense 0.778 vs BM25 0.653 nDCG'@10 (standard nDCG@10: 0.563 vs 0.639). Dense's apparent deficit with keyword queries was largely an artefact of unjudged documents.

**Validity note for the report.** Reusing qrels for reworded queries is limited by pool completeness: 45–60% of the rewrites' top-10 documents are unjudged. Standard nDCG@10 therefore overstates the drops, while judged-only metrics can overstate quality (they assume unjudged documents would have been judged in the same proportion). Agreement across all three measures on the direction of every main effect is what makes the conclusions credible.

Commands:
```
python -m src.stats --dataset trec_covid --measure "<M>" --label rewrite_a_vs_original --pairs bm25_rewrite_a:bm25_query dense_rewrite_a:dense_query rrf_rewrite_a:rrf_query
python -m src.stats --dataset trec_covid --measure "<M>" --label h1 --pairs dense_rewrite_a-dense_query:bm25_rewrite_a-bm25_query ...
```
`<M>` is `nDCG@10`, `nDCG(judged_only=True)@10` or `Bpref`. Outputs: `results/trec_covid/significance_*.csv`.

### 3.8 RQ3 first results: kinds of rewording and query features (25 Oct 2026)

`src/robustness.py`. Each rewrite is one observation (100). Per rewrite: change = score(rewrite) − score(original) for each system, and **dense advantage** = change(dense) − change(BM25). Analysis 1: mean change and mean dense advantage per phenomenon, with bootstrap 95% CI and sign-flip randomization test, Holm across the phenomena with ≥ 10 rewrites. Analysis 2: Spearman correlation of each rewrite feature with the dense advantage, permutation test, Holm across features.

**By phenomenon (nDCG@10 change):**

| Phenomenon | n | BM25 | Dense | RRF | Dense advantage [95% CI] | p_holm nDCG@10 / nDCG'@10 / bpref |
|---|---|---|---|---|---|---|
| paraphrase | 42 | −0.368 | −0.167 | −0.311 | **+0.200** [+0.109, +0.292] | 0.0006 / 0.086 / 0.0003 |
| deanchoring | 34 | −0.337 | −0.122 | −0.288 | **+0.215** [+0.088, +0.345] | 0.0042 / 0.60 / 0.0078 |
| alias | 16 | −0.365 | −0.063 | −0.298 | **+0.302** [+0.127, +0.495] | 0.0049 / 0.060 / 0.026 |
| entity_description | 7 | −0.549 | −0.387 | −0.437 | +0.162 | not tested (n < 10) |
| temporal | 1 | −0.179 | 0.000 | 0.000 | +0.179 | not tested |

Dense advantage on judged-only nDCG'@10: paraphrase +0.087, deanchoring +0.031, alias +0.192, entity_description +0.082 (all positive, none significant after Holm).

**By feature (Spearman ρ with dense advantage, nDCG@10):** content_overlap +0.199 (p = 0.043, p_holm = 0.25), length_change +0.191 (p = 0.056), mean_idf +0.159, n_words +0.137, max_idf +0.128; none significant after Holm. Similar on nDCG'@10 and bpref. `n_acronyms` is 0 for every rewrite (all rewrites were written in lower case), so it carries no information.

Findings:
1. Dense's advantage is positive for every kind of rewording, and significant for the three large groups on nDCG@10 and bpref.
2. **Aliases favour dense most:** dense barely changes (−0.06) when an entity is renamed (veklury, hcq, decadron), while BM25 loses −0.37. This is also the largest judged-only advantage (+0.19).
3. **Entity descriptions hurt every system most** (BM25 −0.55, dense −0.39). Indicative only (n = 7).
4. **No surface feature significantly predicts the dense advantage.** The kind of change matters more than overlap, IDF or length.

Caveat: the two rewrites of a topic are not independent observations; the tests treat the 100 rewrites as exchangeable.

Outputs: `results/trec_covid/rq3_by_phenomenon_*.csv`, `results/trec_covid/rq3_features_*.csv`. Command: `python -m src.robustness --dataset trec_covid [--measure ...]`.

## 4. Pending work
See `docs/ROADMAP.md` for the full plan.

| Step | Component | Notes |
|---|---|---|
| A9 | NevIR (RQ4) | BM25, E5, cross-encoder reference |
| A10 | Error-analysis tool | Top-10 inspection with labels |
| B2 | RQ3 in depth | Per-topic failure examples, ranking stability (top-10 overlap between original and rewrite) |
| B3 | RQ5 complementarity | Relevant documents unique to each system, oracle union recall, why RRF loses its advantage on rewrites |
| B4 | Report | Figures, final report, presentation and demo |
