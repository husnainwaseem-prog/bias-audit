# Model Explainability and Bias Audit

A RandomForest classifier predicts whether a person earns more than
$50K/year on the UCI **Adult Income** dataset (30,162 train / 15,060
test rows). The project explains the model globally and locally with
**SHAP**, measures performance gaps across **sex** and **race**, and
tests a **reweighing** mitigation — reporting how accuracy and
fairness changed.

## Headline results

| Model | Accuracy | Demographic parity diff (sex) | Equal opportunity diff (sex) | Equalized odds diff (sex) |
|---|---|---|---|---|
| Baseline | 0.8463 | 0.1827 | 0.0661 | 0.0835 |
| No sensitive attributes | 0.8438 | 0.1757 | 0.0529 | 0.0783 |
| **Reweighed** | **0.8461** | **0.1294** | **0.0592** | **0.0592** |

- The baseline is accurate but biased: men are predicted high-income
  27.0% of the time vs 8.8% for women (a 0.18 gap).
- **Dropping sex and race barely helps** (0.183 → 0.176): proxies such
  as `relationship=Husband` and marital status carry the signal.
- **Reweighing cuts the gap by 29%** (0.183 → 0.129) at essentially no
  accuracy cost (0.8463 → 0.8461), and brings the race parity gap
  under 0.10 (0.136 → 0.100).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
python -m src.cli audit       # trains 3 models, SHAP, fairness, plots
python -m pytest tests/ -q    # 7 tests
```

## Project structure

```
bias-audit/
├── src/
│   ├── data.py       # load + clean Adult Income (official UCI split)
│   ├── model.py      # RandomForest pipeline (100 trees)
│   ├── explain.py    # SHAP: global importance, beeswarm, waterfall
│   ├── fairness.py   # per-group metrics + gap metrics
│   ├── mitigate.py   # reweighing (Kamiran & Calders)
│   ├── audit.py      # runs baseline -> ablation -> mitigation
│   └── cli.py
├── data/adult.data, adult.test   # UCI Adult Income
├── results/
│   ├── audit.json
│   └── plots/        # shap_importance, shap_beeswarm, 2 waterfalls, fairness_gaps
├── tests/test_audit.py   # 7 tests
├── README.md
└── REPORT.md         # the written audit with conclusions
```

## What SHAP found

Top drivers of a >50K prediction: being married (`Married-civ-spouse`),
age, `relationship=Husband`, capital gains, and years of education.
`sex=Female` is the 9th most important feature and pushes predictions
down — the model uses sex directly, and `relationship=Husband` (3rd)
is a near-perfect proxy for being male.

## Fairness metrics

- **Demographic parity diff**: max gap in selection rate (share
  predicted positive) between groups.
- **Equal opportunity diff**: max gap in true positive rate.
- **Equalized odds diff**: max of the TPR and FPR gaps.
- A common working rule is gaps under 0.10 are small; the audit
  reports the numbers and lets the reader judge.

See `REPORT.md` for the full written audit.
