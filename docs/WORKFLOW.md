# Team Workflow: Tools and Platforms

This project uses three tools:

| What | Where | Why |
|---|---|---|
| **Write code** | VS Code (on your laptop) | Proper editor, git integration, easy to read and review code |
| **Store code** | GitHub (private repo `HamsiniGururaj/robust-news-retrieval`) | One shared copy of the code, with history |
| **Run code** | Google Colab | Free GPU for dense encoding, Linux + Java for Pyserini |

Large data (the corpus, indexes, embeddings) lives in **Google Drive**, never in GitHub.

```
 VS Code (laptop) ──git push──▶ GitHub ──git pull──▶ Colab ──reads/writes──▶ Google Drive
   write & edit                 source of truth        run experiments          data, indexes, embeddings
```

The rule: **code changes happen in VS Code, never in Colab.** Colab only pulls and runs. If you edit a file inside Colab, that change is lost when the session ends.

---

## 1. One-time setup

### GitHub
1. Accept the collaborator invite to the private repo. 
2. Clone it on your laptop:
   ```
   git clone https://github.com/HamsiniGururaj/robust-news-retrieval.git
   ```
3. Create a **personal access token** (GitHub → Settings → Developer settings → Fine-grained tokens) with *read* access to the repo's contents. Colab uses it to clone the private repo.

### VS Code
- Open the cloned folder in VS Code.
- Recommended extensions: **Python**, **Jupyter** (for viewing notebooks).
- You don't need to run the heavy code locally. Small tests on tiny inputs should still work if you have Python installed.

### Google Colab
1. Open Colab → the key icon (**Secrets**) in the left sidebar → add a secret named `GITHUB_TOKEN` with your token as the value. Turn on "Notebook access".
2. Open `notebooks/00_setup.ipynb` from GitHub (File → Open notebook → GitHub tab, tick "Include private repos").

### Google Drive
- The shared data folder is `ir-project-data`. Once it is shared with you, **right-click it → Organize → Add shortcut → My Drive**, so it appears at:
  ```
  /content/drive/MyDrive/ir-project-data
  ```
  The code expects exactly this path. Without the shortcut, it won't find the data.

---

## 2. Everyday workflow

1. **Pull first** in VS Code to get your teammates' changes: `git pull`
2. **Write or edit code** in VS Code. Real logic goes in `src/*.py`. Notebooks only call it.
3. **Commit and push**:
   ```
   git add <files>
   git commit -m "Short description of the change"
   git push
   ```
4. **Run in Colab**: open `notebooks/00_setup.ipynb` and run the cells in order:
   1. Mount Drive
   2. Clone the repo
   3. Install Java + requirements
   4. **Runtime → Restart session** (needed after the first install, otherwise imports such as Pillow fail), then re-run the Drive mount and `%cd /content/robust-news-retrieval`
   5. Sanity check: it should print `All good`
5. Run the experiment notebook or script you need. Outputs that belong in git (small TREC run files, per-topic scores) can be copied back and committed from VS Code. Large outputs stay in Drive.

> **Do not run the last cell of the setup notebook** (`pip freeze > requirements-lock.txt`) unless we have agreed to update the locked package versions. It overwrites the lock file.

If you pushed new code while a Colab session is already open, run `!git pull` inside `/content/robust-news-retrieval` instead of cloning again.

---

## 3. What goes where

| Folder / place | Contents | In GitHub? |
|---|---|---|
| `src/` | Python modules with the real logic (each runnable from the command line) | Yes |
| `notebooks/` | Thin Colab drivers that call `src/` | Yes |
| `configs/` | Paths and settings (no hardcoded paths in code) | Yes |
| `queries/` | Topic files and our query rewrites | Yes |
| `runs/` | TREC-format run files (small text files) | Yes |
| `docs/` | Proposal, reports, this file | Yes |
| Google Drive `ir-project-data/` | Corpus, Lucene indexes, embeddings, FAISS indexes | **No, never** |

**Never commit the Washington Post corpus.** It is controlled-access under a NIST agreement. `.gitignore` already blocks `data/`, `indexes/`, `*.npy` and `*.faiss`, but double-check `git status` before you commit.

Git does not track empty folders, so a folder only appears on GitHub once it contains a file.

---

## 4. Colab tips

- **Colab disconnects** after inactivity or about 12 hours. Long jobs such as dense encoding of the corpus run in **shards** and save progress to Drive, so a rerun resumes where it stopped instead of starting over.
- For dense encoding, choose **Runtime → Change runtime type → GPU**.
- Everything under `/content/` (except the mounted Drive) is wiped when the session ends. Anything you want to keep must be in Drive or pushed to GitHub.
- Environment: Python 3.13, Java 21. Exact package versions are in `requirements-lock.txt`.
