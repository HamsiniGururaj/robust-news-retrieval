# Query Rewrite Guide (RQ2 / RQ3)

How to write the 100 query rewrites (2 per TREC-COVID topic) that the robustness study runs on. Written for a single annotator.

## Why the rewrites matter, and the one rule behind all the others

The experiment asks: **if the same information need is worded differently, how much does each system's score change?** We reuse the NIST relevance judgements for the rewrites. That is only valid if **every rewrite asks for exactly the same documents as the original.** If a rewrite broadens, narrows or flips the need, the old judgements no longer apply and the result is meaningless.

So for every rewrite, the test is:

> Reading the topic's **narrative**, would exactly the same documents be relevant for my rewrite as for the original?

---

## The rules

**Must (errors if broken; the checker enforces the ones it can):**
1. **Same information need.** Check against the narrative, not just the short query.
2. **No negation added** (*not, no, without, except, ...*). It changes relevance; negation is tested separately on NevIR.
3. **No number swaps and no entity substitutions.** "Canada" → "Ontario" or "remdesivir" → "favipiravir" changes the need. Replacing a name with an *equivalent* name or an *unambiguous* description is fine (see phenomena).
4. **Exactly one phenomenon tag** per rewrite, the main change you made (list below).
5. **Rewrite A and rewrite B must differ.**

**Should (warnings; fix, or explain in the `notes` column why not):**

6. **Remove the anchor word** (column `anchor`). It's the original's rarest word, the one BM25 relies on most, so removing it is what makes the rewrite a real test of vocabulary mismatch.
7. **Low overlap.** Share at most about half of the content words with the original. Generic words like "coronavirus" may stay.
8. **Same form and similar length.** A keyword query stays a keyword query, a question stays a question, within ±3 words. *Why:* our first results showed dense retrieval changes by +0.134 nDCG@10 just from keyword → question form. If rewrites changed the form too, we couldn't separate the effect of different words from the effect of a different query style.
9. **A and B use different phenomena** where the topic allows, so each phenomenon gets enough examples.
10. **Natural.** Something a real person could type.

### When the anchor is a generic word
Sometimes the anchor is unremarkable (*die*, *outside*, *impacts*, *spread*). Still replace it: the checker picked it because it's the rarest word *in this corpus*, which is exactly what BM25 relies on. If replacing it would change the meaning, keep it and write `anchor kept: <reason>` in `notes`.

---

## The five phenomena (use these exact tags)

| Tag | What you change | Example on a real topic |
|---|---|---|
| `paraphrase` | Replace a content phrase with a synonym or rewording | 13 *how does coronavirus spread* → **how is SARS-CoV-2 transmitted** |
| `alias` | Acronym ↔ full name, or another established name for the same thing | 20 *coronavirus and ACE inhibitors* → **SARS-CoV-2 and angiotensin-converting enzyme blockers**; 30 *coronavirus remdesivir* → **SARS-CoV-2 GS-5734** (remdesivir's development code) |
| `entity_description` | Replace a named entity with a description that identifies **only** it | 30 *coronavirus remdesivir* → **Gilead antiviral for COVID-19** |
| `temporal` | An equivalent date or period expression | Rare in TREC-COVID. Use only where the topic has a real time element; zero is fine and will be reported |
| `deanchoring` | Drop the anchor (rarest) term and restate the need in other words | 2 *coronavirus response to weather changes* → **COVID-19 seasonality with temperature and humidity** |

These examples show the idea; write your own wording for the sheet. If you do reuse one, say so in `notes`.

### Common mistakes (all of these change the need, so they are not allowed)

| Original | Bad rewrite | Why it's wrong |
|---|---|---|
| 46 *dexamethasone coronavirus* | corticosteroids for COVID-19 | **Broader:** dexamethasone is one of many corticosteroids |
| 3 *coronavirus immunity* | COVID-19 antibodies | **Narrower:** immunity also covers T-cells etc. Check the narrative |
| 9 *coronavirus in Canada* | coronavirus in Ontario | **Entity substitution:** a different place |
| 18 *masks prevent coronavirus* | masks do not stop coronavirus | **Negation:** flips the need |
| 4 *how do people die from the coronavirus* | COVID-19 death causes | **Form change:** question → keyword (warning) |

---

## Step by step

Everything happens in `queries/variants/trec_covid_rewrites.csv`: one row per topic, already filled with `qid`, `original`, `question`, `narrative` and `anchor`. You fill `rewrite_a`, `phenomenon_a`, `rewrite_b`, `phenomenon_b`, `verified`, `notes`.

**Editing the CSV:**
- **Excel:** double-click to open. When saving, use **File → Save As → "CSV UTF-8 (Comma delimited)"**.
- **Google Sheets:** File → Import, then **File → Download → CSV**, and replace the file in the repo.
- **VS Code:** an extension such as *Edit CSV* gives a table view.

Don't rename, add or delete columns or rows, and don't edit `original`. The checker always uses the official original anyway.

### Day 1: write (about 3–4 hours in total)
1. **Pass 1, rewrite A for all 50 topics** (~2 min each). For each row:
   1. Read `original`, `question` and `narrative` to understand exactly what is relevant.
   2. Look at `anchor`.
   3. Pick the phenomenon that fits this topic best: alias if an established alias exists, entity description for a named drug or entity, otherwise paraphrase or de-anchoring.
   4. Write the rewrite and the tag.
2. **Pass 2, rewrite B for all 50**, using a different phenomenon from A where possible.
3. **Run the checker** often while you work. It shows exactly which rows need attention:
   ```
   python -m src.variants check --dataset trec_covid
   ```
   - **ERROR:** must be fixed.
   - **warning:** fix it, or explain in `notes` why it's acceptable.

### Day 2: verify (about 1 hour)
Since there is no second annotator, verification is a **separate pass on a different day**, done as blind as possible:

1. In the sheet, **hide the `original` and `question` columns.**
2. For each rewrite, read **only the narrative and the rewrite**, and ask: *would exactly the same documents be relevant?*
3. If yes, write `yes` in `verified`, once both of that topic's rewrites pass. If not, fix the rewrite and note what changed.
4. Run the checker again. With no errors, it writes:
   - `queries/trec_covid_rewrite_a.tsv`, `queries/trec_covid_rewrite_b.tsv`: the query files for the runs
   - `queries/variants/trec_covid_rewrite_features.csv`: overlap, anchor removed, IDF, length, numbers and acronyms per rewrite, for RQ3

### Then
5. Commit the sheet, the two TSVs and the features CSV.
6. Run the systems on the rewrites in Colab (`notebooks/01_pipeline.ipynb`, Part 4).

## Two rules that protect the results
- **Finish and verify all rewrites before running any system on them.**
- **Don't edit rewrites after seeing their scores.** Tuning wording to make a system win or lose would bias the study. If you find a genuine *validity* problem later (the meaning changed), fix it, write `changed after results: <reason>` in `notes`, and mention it in the report.

## How to describe this in the report
> Two meaning-preserving rewrites were written for each of the 50 topics by a single annotator, following a fixed rubric (one phenomenon per rewrite; no negation, number or entity changes; same query form and similar length; the topic's rarest term removed where possible). Each rewrite was checked automatically (negation, overlap, anchor-term removal, length and form) and then re-verified against the topic narrative in a separate pass on a different day, without viewing the original query. Single-annotator verification is a limitation; the proposal planned independent verification by a second group member.
