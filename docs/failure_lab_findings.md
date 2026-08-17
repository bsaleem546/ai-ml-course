# Failure Lab Findings — Stage 3

Seven failure modes deliberately induced and diagnosed against the Telco Customer Churn
dataset, using a `DecisionTreeClassifier` throughout so complexity, capacity, and structure
could be varied while holding the model type constant. All code lives in
`scripts/failure_lab.py`; all numbers below were actually produced by running it, not
theoretical.

Baseline for comparison throughout: `DummyClassifier(strategy="most_frequent")` — accuracy
0.7345, precision/recall/F1 all 0.0 (never predicts churn). A properly-tuned honest model
(`max_depth=5`, natural class balance, no leakage): accuracy 0.7946, precision 0.6152, recall
0.6043, F1 0.6097, ROC-AUC 0.8290.

---

## 1. Overfitting

**Symptom:** train accuracy 0.9980, val accuracy 0.7293 — a 26.87-point gap. Val accuracy is
actually *worse* than the majority-class baseline (0.7345) despite near-perfect training
performance.

**Root cause:** `DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)` — no complexity
limit, so the tree kept splitting until it effectively memorized individual training rows
instead of learning a generalizable pattern.

**Fix:** cap model complexity. A `max_depth` sweep (`[1, 2, 3, 5, 10, 20, None]`) showed val
accuracy peaking at **`max_depth=5`** (0.7946) with a small gap (0.0078), then declining as
depth increased further (gap balloons to 0.27 by `max_depth=20`/unbounded). The complexity
that maximizes *validation* performance, not training performance, is the right choice.

---

## 2. Underfitting

**Symptom:** `max_depth=1` — train accuracy 0.7347, val accuracy 0.7345, gap ~0.0002.
Consistent between train and val, but both numbers are mediocre — barely above the baseline.

**Root cause:** a 1-split tree doesn't have enough capacity to capture the real pattern in
the data at all, regardless of which set it's tested on.

**Fix:** more model capacity (see `max_depth=5` above) or better input signal. Tested
feature engineering (`NumServices`, `AvgMonthlyCharge`) as an alternative fix — at
`max_depth=1` it made **zero difference** (identical accuracy), because a 1-split tree can
only ever use one feature regardless of how many good ones are available. At `max_depth=3`,
still zero difference — these two engineered features turned out to be fully redundant with
information the tree already had access to via its component columns. **Key lesson:** a
small train/val gap alone doesn't mean "good" — it means "consistent." Need both a small gap
*and* genuinely good scores.

---

## 3. Target leakage

**Symptom:** adding a synthetic feature (`CancellationRequestFiled`, ~95% correlated with
the true `Churn` label) pushed val accuracy from 0.7946 to **0.9560** — a 16-point jump, at
identical model complexity (`max_depth=5`).

**Root cause:** `CancellationRequestFiled` modeled a realistic bug — a field that only gets
populated *because* a customer is already churning (e.g. they filed a cancellation request),
so it wouldn't genuinely be available before the outcome it's supposed to predict.

**Detection:** two independent methods both caught it. (1) The suspiciously large accuracy
jump itself. (2) Feature importance — `CancellationRequestFiled` alone accounted for **92.1%**
of the leaky model's total importance, vs. the honest model's top feature only reaching 51%.

**Fix:** removed the feature, accuracy returned to exactly 0.7946, confirming it was the sole
cause. **General defense:** leakage is invisible to train/val-gap checks and cross-validation
(the leak is present equally in both splits) — the only real defense is asking *"would this
feature genuinely be available before the prediction is needed, in production?"*, not a
metric-based check.

---

## 4. Class imbalance

**Symptom:** training on an artificially extreme 5% churn rate (down from the natural 26.5%,
by subsampling churned customers in the training set only) produced accuracy 0.7397 (looks
fine) but **recall 0.0250** — the model caught only 14 of 561 actual churners in the
validation set (confusion matrix: TN 1549, FP 3, FN 547, TP 14).

**Root cause:** with only 190 churned training examples (down from 1308), the model had too
little signal to learn what a churner looks like, and defaulted toward almost always
predicting the majority class.

**Detection:** accuracy alone completely hid this (0.7397 looks unremarkable). Precision was
actually the *highest* of all models tested (0.8235) — when it did predict churn, it was
usually right, it just almost never did. F1 (0.0484) was the metric that actually exposed the
problem, punished hard by the near-zero recall.

**Fix:** ROC-AUC revealed something important — the imbalance-trained model's AUC (0.8003)
was nearly as good as the honest model's (0.8290), meaning its underlying *ranking* ability
was largely intact. The real problem was *calibration* (it learned to output uniformly lower
probabilities), not a fundamental inability to distinguish churners — making it a strong
candidate for threshold tuning (see below) as a retrain-free mitigation, distinct from fixing
the training data itself.

---

## 5. Threshold sensitivity

**Symptom:** sweeping the decision threshold from 0.1 to 0.9 on the honest model showed
precision ranging 0.386→1.000 and recall ranging 0.964→0.002 — the same trained model,
wildly different behavior depending on the cutoff applied to its output probabilities.

**Root cause:** `.predict()`'s default 0.5 cutoff is an arbitrary convention, not something
inherent to the model. `predict_proba()` gives the real, continuous signal; `.predict()` is
just one specific way of converting that signal into a hard decision.

**Finding:** F1 peaked near threshold 0.3 (0.6076), marginally above the default 0.5 (0.6070)
— a shallow, broad plateau from ~0.2-0.5, then a sharp drop-off past 0.5. Also observed:
thresholds 0.6 and 0.7 gave *identical* results, because a decision tree only outputs a
handful of distinct leaf probabilities — threshold changes within a gap between two leaf
values have zero effect (a limitation specific to trees, unlike logistic regression's smooth
probability output).

**Fix/takeaway:** the right threshold depends on the real cost of each mistake (a missed
churner vs. a wasted retention offer) — there's no universal correct value, and it can be
tuned without retraining.

---

## 6. Ranking vs. calibration (ROC-AUC)

**Symptom:** ROC-AUC comparison — baseline 0.5000 (confirms zero ranking ability), honest
0.8290, 5%-imbalance-trained 0.8003, overfit (unbounded depth) 0.6526.

**Root cause distinction this revealed:** two very differently-broken models can look
similarly bad on accuracy/precision/recall/F1 alone, because those all depend on one
committed threshold. ROC-AUC separates "does the model fundamentally understand the pattern"
(ranking quality across all thresholds) from "is it making good decisions at this specific
cutoff" (calibration). The imbalance-trained model's ranking was nearly as good as the honest
model's (0.80 vs 0.83) — its problem was calibration, fixable via threshold tuning. The
overfit model's ranking was genuinely much weaker (0.65) — a deeper problem, not fixable by
adjusting a threshold, because it never learned a transferable pattern in the first place.

---

## 7. Distribution shift

**Symptom:** simulating a 20% price increase (`MonthlyCharges`/`TotalCharges` scaled up on
validation data, labels held fixed) degraded the honest model from accuracy 0.7946→0.7743,
**precision 0.6152→0.5709** (biggest drop), recall nearly flat (0.6043→0.6025), F1
0.6097→0.5863.

**Root cause:** the model's `StandardScaler` and the tree's learned split thresholds were
calibrated to the *original* charge distribution. Feeding systematically shifted values
pushed more customers across those learned boundaries, causing the model to flag churn more
often overall — catching roughly the same real churners (flat recall) but with more false
alarms (precision drop).

**Why this is the most dangerous failure mode covered:** unlike overfitting (visible train/val
gap) or leakage (suspiciously perfect numbers), distribution shift produces quiet, gradual
decay with no obvious single-snapshot red flag. It's only catchable by monitoring performance
over time against a stable baseline — which is why production ML systems need drift detection
and periodic retraining, not a train-once-and-forget approach.

---

## Cross-cutting lessons

- **No single metric tells the whole story.** Accuracy hid the imbalance failure; the
  train/val gap alone wouldn't have caught leakage without also checking magnitude; ROC-AUC
  was needed to distinguish a calibration problem from a ranking problem. Real ML evaluation
  needs multiple, complementary diagnostics, not one number.
- **A small train/val gap means "consistent," not "good."** Underfitting has a near-zero gap
  and is still a failure.
- **Feature importance is an independent detection method**, not just an explainability
  nicety — it caught the same leakage bug the accuracy jump did, via a completely different
  signal (one feature dominating 92% of importance).
- **Threshold tuning and retraining are different tools for different problems** — the former
  fixes calibration without touching the model; the latter is needed when the model's
  underlying ranking ability itself is weak.
