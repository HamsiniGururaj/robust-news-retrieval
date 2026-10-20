# Robust Retrieval Under Vocabulary Mismatch

Information Retrieval course project, Group 5.

A controlled robustness study of **BM25**, a **dense bi-encoder (E5-base-v2)** and a **hybrid (reciprocal rank fusion)** when the same information need is reworded. The corpus and the information need stay fixed; only the query wording changes.

- **Primary benchmark:** TREC-COVID (BEIR version). It replaces the proposed TREC 2018 Common Core / Washington Post collection, which could not be obtained in time. See [docs/PROGRESS.md](docs/PROGRESS.md) for the reasoning.
- **Negation diagnostic:** NevIR, reported separately.
- **Pipeline smoke test:** BEIR SciFact.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | All logic, as modules with command-line entry points |
| `configs/` | Settings: `default.yaml` (fixed system settings) plus one file per dataset |
| `queries/` | Exported topics and qrels; our query rewrites |
| `runs/` | TREC-format run files |
| `results/` | Per-topic evaluation scores |
| `notebooks/` | Thin Colab drivers that call `src/` |
| `tests/` | Unit tests on tiny inputs |
| `docs/` | Proposal, progress log, roadmap, team workflow |

Large files (corpora, indexes, embeddings) live in Google Drive, never in git.

## Quick start

Local (lightweight steps and tests):
```
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest -q
```

Export topics and qrels, then evaluate run files:
```
python -m src.load_topics --dataset trec_covid
python -m src.evaluate --dataset trec_covid --runs runs/trec_covid/*.run
```

Print the settings a run will use:
```
python -m src.config --dataset trec_covid
```

Heavy steps (indexing, dense encoding) run in Google Colab via `notebooks/01_pipeline.ipynb`. See [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Documentation
- [docs/proposal.pdf](docs/proposal.pdf): original proposal
- [docs/PROGRESS.md](docs/PROGRESS.md): what's built, dataset change and decisions
- [docs/ROADMAP.md](docs/ROADMAP.md): implementation plan
- [docs/WORKFLOW.md](docs/WORKFLOW.md): VS Code → GitHub → Colab workflow
