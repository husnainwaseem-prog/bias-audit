# Model Explainability and Bias Audit — Report

## 1. Task

Take a classification model on a tabular dataset with sensitive
attributes, explain its predictions globally and locally with SHAP,
measure performance gaps across groups, apply at least one mitigation
technique, and report how accuracy and fairness metrics changed.

## 2. Dataset and model

**Dataset.** UCI Adult Income (Census): 30,162 training rows and
15,060 test rows (the official split). Target: income > $50K/year
(24.9% positive in train). Sensitive attributes: **sex** (Male/Female)
and **race** (White/Black/Asian-Pac-Islander/Amer-Indian-Eskimo/Other,
grouped to White/Black/Other for the tables). Rows missing workclass,
occupation, or native-country were dropped (the standard treatment).

**Model.** RandomForest, 100 trees, one-hot encoded categoricals.
Test accuracy **0.8463**. The same pipeline is used for all three
variants so the comparison is apples-to-apples.

## 3. What the model learned (SHAP)

SHAP values computed with `shap.TreeExplainer` on 100 test rows
(exact values; 100 rows keep the runtime sane — each row costs
~1.7 s on this hardware because the trees are deep).

**Global — mean |SHAP| (top 5):**

| Feature | mean \|SHAP\| | Direction |
|---|---|---|
| marital-status=Married-civ-spouse | 0.058 | pushes up |
| age | 0.046 | older pushes up |
| relationship=Husband | 0.044 | pushes up |
| capital-gain | 0.042 | higher gains push up |
| education-num | 0.040 | more schooling pushes up |

`sex=Female` ranks 9th (0.012) and pushes predictions **down**; `sex=Male`
pushes up. The model uses sex directly. Just as important,
`relationship=Husband` (3rd) is a near-perfect proxy for being male —
so even without the sex column, the model can reconstruct it.

**Local — waterfall for one person** (`results/plots/shap_waterfall_0.png`):
a 35-year-old married man in tech support, predicted 0.79 probability
of >50K from a base rate of 0.25. The biggest pushes: Tech-support
occupation (+0.11), married (+0.09), Husband (+0.07). A second waterfall
shows a contrasting low-probability case. Both plots are in `results/plots/`.

## 4. Fairness gaps (baseline)

| Group | n | Base rate (>50K) | Accuracy | Selection rate | TPR | FPR |
|---|---|---|---|---|---|---|
| Female | 4,913 | 0.113 | 0.926 | 0.088 | 0.560 | 0.027 |
| Male | 10,147 | 0.310 | 0.808 | 0.270 | 0.626 | 0.111 |
| Black | 1,411 | 0.119 | 0.923 | 0.090 | 0.554 | 0.027 |
| White | 12,970 | 0.260 | 0.838 | 0.226 | 0.621 | 0.087 |

**Gaps (baseline):** sex — demographic parity 0.183, equal opportunity
0.066, equalized odds 0.084. Race — demographic parity 0.136, equal
opportunity 0.068, equalized odds 0.068.

Two honest caveats. First, the base rates genuinely differ (11% of women
vs 31% of men earn >50K in this 1994 data), so part of the selection-rate
gap reflects the data, not the model — which is why equal opportunity
(TPR gap, 0.066) is the more meaningful number: among people who *do*
earn >50K, the model finds 62.6% of men but only 56.0% of women.
Second, accuracy is *higher* for women (0.926 vs 0.808) precisely because
the model rarely predicts them positive — accuracy alone would hide the
problem, which is why the audit reports rates, not just accuracy.

## 5. Mitigation

**Attempt 1 — drop the sensitive attributes.** Accuracy 0.8438
(-0.0025). Sex parity gap 0.183 → 0.176: barely moved. The proxies
(`relationship=Husband`, marital status, occupation, hours-per-week)
carry almost all of the signal. Removing the columns does not remove
the bias. This is the audit's most important negative result.

**Attempt 2 — reweighing (Kamiran & Calders, 2012).** Each training
example is weighted so sex/race and the label become independent in
the weighted training set: `w(group, label) = P(group)·P(label) /
P(group, label)`. No model change, no threshold hacking.

| Metric | Baseline | Reweighed | Change |
|---|---|---|---|
| Accuracy | 0.8463 | 0.8461 | −0.0002 |
| Demographic parity diff (sex) | 0.1827 | 0.1294 | −29% |
| Equal opportunity diff (sex) | 0.0661 | 0.0592 | −10% |
| Equalized odds diff (sex) | 0.0835 | 0.0592 | −29% |
| Demographic parity diff (race) | 0.1355 | 0.0996 | −27% |
| Equal opportunity diff (race) | 0.0678 | 0.0220 | −68% |
| Equalized odds diff (race) | 0.0678 | 0.0322 | −53% |

Reweighing is the clear winner: the accuracy cost is negligible
(-0.0002) while every fairness gap shrinks, and the race parity gap
falls under the 0.10 rule-of-thumb. The sex parity gap (0.129) is
still above 0.10 — reweighing reduces the bias, it does not eliminate
it, because the underlying base-rate difference remains.

## 6. Conclusions

1. **The model is biased, and accuracy hides it.** 84.6% accuracy
   coexists with a 0.18 selection-rate gap by sex and a 0.07 TPR gap.
2. **Fairness through unawareness fails.** Dropping sex and race
   leaves the gap almost untouched — proxies reconstruct them.
3. **Reweighing works and is cheap.** A 29% cut in the parity gap
   for a 0.0002 accuracy cost. It is the mitigation to ship.
4. **The remaining gap is partly in the data.** Even a perfectly
   calibrated model would show a selection-rate gap given the 11% vs
   31% base rates; the TPR gap (0.059 after mitigation) is the fairer
   target, and it is the one to keep monitoring.
5. **Next steps:** per-group threshold tuning (post-processing) could
   push the TPR gap lower; collecting fresher data would address the
   base-rate gap itself; and any deployment should re-run this audit
   on every model update.

## 7. How to reproduce

```bash
pip install -r requirements.txt
python -m src.cli audit        # ~6 min: 3 model fits + SHAP on 100 rows
python -m pytest tests/ -q     # 7 tests
```

Note: SHAP's exact TreeExplainer costs ~1.7 s per row on these deep
trees, so the audit uses 100 rows for the SHAP plots (the `--shap-n`
flag controls this). The fairness numbers use the full test set.
