# SoilBin V29.1 — Generalization Reliability Meta-Learning

Status: executed as nested Speed×Load-group OOF development analysis.

## Four requested modules

1. Error Predictor
- Predict absolute unseen error for each expert.
- Experts: V23, Hierarchical, Reverse-Stack.
- Low-capacity Ridge models.
- Input features are available before the target plus cross-fitted expert predictions and training-only diagnostics.

2. Winner Classifier
- Multiclass Logistic Regression.
- Target: expert with smallest absolute error in inner held-out groups.
- Outer-test labels are never used for classifier training.

3. Generalization-Gap Predictor
- Predict unseen absolute error minus the expert's training-side pass error.
- Converts training-side performance into an expected unseen-error estimate.

4. Meta-Router
- Fixed rank aggregation of Error Predictor, Winner Classifier, and Gap Predictor.
- No extra outer-test tuning.
- Tie-breaker: Error Predictor.

## Validation guards
- Whole Speed×Load group held out.
- Inner group cross-fitting builds all meta-training examples.
- No random row split.
- No current/future target used as a meta feature.
- Outer-test labels used only after predictions are frozen for evaluation.
