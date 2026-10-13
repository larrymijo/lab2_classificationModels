# Lab 2 — Classification Models

**DA380A Machine Learning · Högskolan Kristianstad · HT2026**
Group: 28 · Members: Lawrence Arciniegas, Gunnar Arias

Practise the complete classification workflow: explore a dataset, prepare it, train at least two
classification models, evaluate them with accuracy / precision / recall / F1 and a confusion
matrix, tune them with cross-validation, and recommend the best model.

## Key dates

| Date | What |
|---|---|
| Tue 13 Oct 2026 | Lab session |
| ~20 Oct 2026 | Deadline (one week after the session; exact time on Canvas) |

**Deliverables:** one documented notebook + one PDF report, uploaded to Canvas.

## Dataset

| | |
|---|---|
| Chosen dataset | `CarData` (`ALL_CARS_DATA.csv`) — see [docs/step1-dataset-choice.md](docs/step1-dataset-choice.md) |
| Target (class) variable | `PRICE RANGE` — a car's price tier compared with other cars of the same type |
| Number of classes / balance | 4 (Low, Medium, High, Luxury), balanced at 24–27% each |

Candidates: `perfumes`, `NetworkEvent`, `musicdata`, `CarData`.

## How to run

1. Put the Canvas datasets in `data/`, one folder each (`data/CarData/ALL_CARS_DATA.csv` is the one used).
2. Open `notebooks/lab2_classification.ipynb`, select the `.venv` kernel, and **Run All**.
3. A full run takes about 5–7 minutes (cross-validation and the random-forest grid search). Lower `SAMPLE_SIZE` in the setup cell while experimenting.

Cleaning lives in `src/data.py`, dataset comparison helpers in `src/overview.py`. The report outline with all results is in `reports/report-outline.md`.

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd lab2_classificationModels
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
nbstripout --install
```

1. Download the four datasets from Canvas and put each in its own folder in `data/` (they are not committed).
2. Open the folder in VS Code, select the `.venv` interpreter, open a notebook in `notebooks/`.

## Project structure

| Folder | Contents |
|---|---|
| `data/` | Datasets from Canvas (git-ignored) |
| `docs/` | Planning and decisions |
| `notebooks/` | Jupyter notebooks |
| `reports/` | Report outline and the PDF report |
| `src/` | Reusable Python helpers |

## Plan

- [x] **Step 0 — Setup:** repo, virtual environment, dependencies
- [x] **Step 1 — Choose dataset:** compare all four, pick one, define the class target and count the classes
- [x] **Step 2 — Explore:** structure, missing values, duplicates, unusual values, **class imbalance**, differences between classes, observations
- [x] **Step 3 — Prepare & split:** clean, define X / y, **stratified** 80/20 train-test split, preprocessing pipeline
- [x] **Step 4 — Models:** baseline + at least two classifiers, compared with 5-fold CV on the training set
- [x] **Step 5 — Evaluate:** accuracy, precision, recall, F1 and a confusion matrix per model on the same test set (ROC-AUC if binary)
- [x] **Step 6 — Tune:** GridSearchCV with 5-fold CV, before/after comparison
- [x] **Step 7 — Interpret:** which classes are easy/hard, strengths/weaknesses, recommendation
- [ ] **Report:** PDF answering the three required questions
- [ ] **Submit** notebook + PDF on Canvas

## Workflow rules

- Don't commit directly to `main` after setup: use a branch per task and open a pull request.
- Only one person edits a given notebook at a time.
- The test set is not used until the final evaluation (Step 5).
- `random_state=42` everywhere so results are reproducible.

## Use of AI tools

AI tools may be used for ideas, explanations and debugging, but not to copy work directly.
Every group member must be able to explain all code and text submitted.
