# M07 RSNA External Failure Analysis and Literature Review — 2026-10-01

## Scope

This note separates the canonical untouched external validation result from all secondary no-base-retraining adaptation analyses. The canonical M07 R224 external result remains the primary reference.

## Canonical project evidence

Expanded RSNA pediatric cohort: 1,099 images / 553 patients.

R224 canonical:
- Precision positive: 0.7311178248
- Sensitivity: 0.9289827255
- Specificity: 0.6920415225
- AUROC: 0.9255557253
- AUPRC: 0.9280665125
- TN / FP / FN / TP: 400 / 178 / 37 / 484

Primary program target:
- Precision >= 0.85 AND Sensitivity >= 0.90
- The target is not met by the canonical operating point.

## Training-domain mismatch

The Guangzhou/Kermany pediatric pneumonia source dataset is composed of AP chest radiographs from children aged 1–5 years. This is a narrow age and acquisition domain.

References:
- Kermany DS et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018. DOI: 10.1016/j.cell.2018.02.010. https://pubmed.ncbi.nlm.nih.gov/29474911/
- Padash S et al. Pediatric chest radiograph interpretation: how far has artificial intelligence come? Pediatr Radiol. 2022. DOI: 10.1007/s00247-022-05368-w. https://pubmed.ncbi.nlm.nih.gov/35460035/
- Public dataset description: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Age audit: age shift is not sufficient to explain the failure

Frozen R224 subgroup results:
- Age 1–5: n=73, precision=0.44898, sensitivity=0.88, specificity=0.4375, AUROC=0.76667
- Age 6–9: n=209, precision=0.72, sensitivity=0.97297, specificity=0.57143, AUROC=0.94337
- Age 10–18: n=817, precision=0.76458, sensitivity=0.91948, specificity=0.74769, AUROC=0.92965

Interpretation:
- The training-age-matched subgroup is the worst-performing age subgroup.
- Therefore simple age mismatch is not supported as the sole or dominant explanation.
- The age 1–5 subgroup is small, so uncertainty remains material.
- Age should remain a stratification variable, not be treated as a causal diagnosis of the failure.

## Projection/view audit: strongest observed failure mechanism

Frozen R224:
- AP: n=599, precision=0.76773, sensitivity=0.99540, specificity=0.20122, TN/FP/FN/TP=33/131/2/433
- PA: n=500, precision=0.52041, sensitivity=0.59302, specificity=0.88647, TN/FP/FN/TP=367/47/35/51

Of the 178 canonical false positives:
- AP: 131 (73.6%)
- PA: 47 (26.4%)

Interpretation:
- AP images are shifted strongly toward positive predictions: near-complete sensitivity but severe false-positive inflation.
- PA images show the opposite operational pattern: high specificity with a large sensitivity collapse.
- This is consistent with a projection-dependent score/operating-point shift, not a single global threshold error.

The external RSNA source itself contains both AP and PA studies and was built from the NIH chest radiograph collection with expert opacity annotations. The endpoint is pulmonary opacity that may represent pneumonia in the appropriate clinical setting, not a perfect etiologic pneumonia diagnosis.

Reference:
- Shih G et al. Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. Radiol Artif Intell. 2019. DOI: 10.1148/ryai.2019180041. https://pmc.ncbi.nlm.nih.gov/articles/PMC8017407/

## Literature support for projection/acquisition shortcut risk

AP and PA projections are so visually separable that a deep network can classify them with near-perfect discrimination, including in pediatric images. That makes projection a plausible shortcut variable if disease prevalence, severity, equipment, or workflow differ by view.

References:
- Kim TK et al. Deep Learning Method for Automated Classification of Anteroposterior and Posteroanterior Chest Radiographs. 2019. https://pubmed.ncbi.nlm.nih.gov/30972585/
- Projection-Related Bias in the Detection of Thoracic Abnormalities: A Large-Scale Analysis of the NIH ChestX-Ray14 Dataset. J Imaging. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13207601/

The 2026 projection-bias analysis reports substantial associations between projection and thoracic findings and explicitly cautions that projection can reflect both technical effects and clinical-severity confounding. This supports treating projection as a heterogeneity marker, not as a proven causal mechanism.

## Literature support for external generalization failure

A particularly relevant published experiment trained a pediatric pneumonia model on Guangzhou children aged 1–5 and evaluated it externally. Internal AUC was 0.95, while external AUC fell to 0.54; activation maps also became less clinically relevant on the external data.

Reference:
- Xin KZ, Li D, Yi PH. Limited generalizability of deep learning algorithm for pediatric pneumonia classification on external data. Emerg Radiol. 2022. DOI: 10.1007/s10140-021-01954-x. https://pubmed.ncbi.nlm.nih.gov/34648114/

More broadly, cross-hospital pneumonia work has shown that chest-X-ray CNNs can learn institution-specific signals and lose performance when moved between health systems.

Reference:
- Zech JR et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. PLoS Med. 2018. DOI: 10.1371/journal.pmed.1002683. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683

A 2022 systematic review found that most pediatric chest-radiograph AI studies were not externally validated. A 2026 pediatric imaging review continues to emphasize age-aware, context-specific validation and domain-shift risk.

References:
- Padash S et al. Pediatr Radiol. 2022. https://pubmed.ncbi.nlm.nih.gov/35460035/
- Muringathuparambil JJ et al. Artificial intelligence in paediatric chest imaging: applications, challenges, and future directions. Pediatr Radiol. 2026. DOI: 10.1007/s00247-026-06671-6. https://doi.org/10.1007/s00247-026-06671-6

## Calibration and no-base-retraining experiments

These are secondary external adaptation analyses, not untouched external validation.

Observed results:
- Cross-fit sensitivity-90 threshold: precision=0.78239, sensitivity=0.90403, FP=131, FN=50
- Equal-resolution cross-fit: precision=0.79661, sensitivity=0.90211, FP=120, FN=51
- Confidence-gated advanced ensemble: precision=0.79730, sensitivity=0.90595, FP=120, FN=49
- Rank-centered advanced ensemble: precision=0.79595, sensitivity=0.90595, FP=121, FN=49
- Fold-consensus cross-fit: precision=0.77250, sensitivity=0.90595, FP=139, FN=49
- View-threshold cross-fit (train floor 0.92): precision=0.75758, sensitivity=0.91171, FP=152, FN=46
- Age×view cross-fit (train floor 0.94): precision=0.75196, sensitivity=0.91939, FP=158, FN=42
- Final cross-fit combination: precision=0.82509, sensitivity=0.89635, FP=99, FN=54
- Beta calibration: precision=0.88382, sensitivity=0.81766, FP=56, FN=95
- Isotonic calibration: precision=0.87841, sensitivity=0.80422, FP=58, FN=102
- Platt calibration: precision=0.86519, sensitivity=0.82534, FP=67, FN=91

Interpretation:
- Post-hoc calibration can sharply increase precision, but in this cohort it does so by sacrificing too much sensitivity.
- Multi-resolution/view-aware cross-fit methods preserve sensitivity better and reduce false positives substantially, but still do not satisfy precision >=0.85 and sensitivity >=0.90 simultaneously.
- The final combination reduces FP from 178 to 99, but observed sensitivity is 0.89635, just below the target floor.
- This pattern suggests that the remaining error is not merely a one-dimensional global calibration defect.

Recalibration can improve probability calibration in chest radiograph models, but this does not guarantee recovery of a desired clinical operating point under domain shift.

Reference:
- Automated identification of chest radiographs with referable abnormality with deep learning: need for recalibration. Eur Radiol. 2020. https://pubmed.ncbi.nlm.nih.gov/32661584/

## TTA and lung-focused inference

The tested deterministic intensity TTA and lung-focused crops did not recover the target:
- Equal TTA slightly worsened precision/FP.
- The best single TTA view improved precision only modestly.
- Lung crops increased sensitivity in some settings but worsened false-positive burden.
- Episodic head-normalization TTA reproduced the canonical confusion matrix and did not improve the result.

Therefore these methods are not candidates for the final combined operating point.

## Current synthesis

Evidence supports the following ordering of hypotheses:

1. Projection/acquisition-dependent operating-point shift: strongly supported by project subgroup evidence and external literature.
2. General dataset/institution/domain shift: strongly supported.
3. Endpoint mismatch (pneumonia vs radiographic lung opacity): important and structurally unavoidable in interpreting the external result.
4. Age distribution shift: present, but not sufficient as the primary explanation because the training-age-matched subgroup performs worst.
5. Pure global threshold/calibration error: only partial explanation; calibration improves precision at unacceptable sensitivity cost.
6. Simple lung-background shortcut recoverable by crop/TTA: not supported by the tested methods.

## Scientific status

The current evidence does **not** demonstrate that M07 is intrinsically poor at ranking external cases: AUROC/AUPRC remain high. The dominant failure is transport of the operating point and class-conditional score distribution across heterogeneous acquisition/view contexts.

The canonical external result must remain unchanged. Any calibrated, view-aware, ensemble, or stacking result must be reported as a secondary external adaptation analysis with explicit leakage controls.

## Pending

The label-free R224 domain-harmonization lane is still executing. Its result should be appended here after the terminal receipt is verified.
