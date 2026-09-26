# Group-Level Uncertainty under Real Label Noise

*Anonymous code repository accompanying a double-blind submission. Do not add author, institution,
or account information to this repository while it is under review.*

A cautionary, reproducible study of whether **group-level** (entity / taxonomy) uncertainty can
localize real-world label noise when individual prediction confidence fails. We implement
**GUARD** (Group Uncertainty And Residual Decomposition), a two-axis group metric, validate it on
controlled synthetic data, and stress-test it on the **AlleNoise** benchmark (real, human,
instance-dependent label noise).

**Headline result (honest):** on real large-scale text noise, group-level claim–belief divergence
does not beat a trivial per-item aggregation baseline. The contribution is the phenomenon it
quantifies, the failure mode it diagnoses and repairs, and the conditions under which the idea
would work.

---

## Key findings

1. **Individual confidence collapses under semantic noise.** Detecting real misregistrations from
   individual signals is near chance (AUROC ≈ 0.53), and it *degrades monotonically* as the
   classifier is trained better (AUROC of `1−p` falls 0.556 → 0.513 as clean accuracy rises
   0.31 → 0.58). Confident-learning / cleanlab scores (0.522–0.526), training-dynamics detectors
   (AUM 0.523; Data-Maps 0.54–0.55), a deep ensemble, and temperature scaling all leave this
   near chance — the failure is a property of the noise, not of the detector.
2. **Group claim–belief divergence saturates in a large label space, and we repair it.** The
   global Jensen–Shannon term is dominated by belief mass outside the subtree (mean ≈ 0.93–1.0);
   a principled subtree projection removes the saturation (mean ≈ 0.09).
3. **Even repaired, GUARD does not beat trivial aggregation.** Ranking entity groups by true noise
   rate, all scores fall in a narrow band (Spearman 0.12–0.21, AUROC 0.58–0.62, overlapping 95%
   CIs); the best is a baseline (the 90th percentile of per-item mismatch), and the GUARD–baseline
   AUROC gap is +0.001 (95% CI [−0.085, +0.090]).
4. **The metric is exact but does not transfer.** In synthetic data the identity `H(p̄)=W+D` holds
   to machine precision and the direction term separates contamination regimes perfectly (AUROC
   1.0). Restricting the real label space or sharpening beliefs moves GUARD toward the baseline
   exactly as predicted, delineating *when* the idea works (small label spaces, sharp classifiers).

### Numbers at a glance
| Aspect | Value |
|---|---|
| Dataset | AlleNoise — 502,310 items · 5,691 categories · 14.75% real noise |
| Classifier | transformer, 3-fold cross-fitted, held-out clean top-1 ≈ 0.553 |
| Individual detection AUROC | 0.51–0.54 (all confidence / CL / AUM / ensemble variants) |
| Group ranking (475 groups) | best baseline p90 → AUROC 0.622; best GUARD → 0.623 (tied) |
| Synthetic sanity | identity error < 2e-16; direction term AUROC 1.0 vs 0.5 aggregation |

---

## Repository structure

```
.
├── requirements.txt
├── src/guard_core.py                  # the GUARD metric (numpy-only, importable)
├── notebooks/
│   ├── 00_download_data.ipynb              # download + profile AlleNoise
│   ├── 01_finetune_and_predict.ipynb       # k-fold cross-fitting -> out-of-sample p_i  [GPU]
│   ├── 02_layer1_detection.ipynb           # individual misregistration detection
│   ├── 03_layer2_signature.ipynb           # initial group construction + noise anchor
│   ├── 04_nontriviality_rq2.ipynb          # structured vs diffuse separability
│   ├── 05_routing_policy.ipynb             # first-pass review-budget routing
│   ├── 06_taxonomy_group_analysis.ipynb    # taxonomy projection + decisive comparison
│   ├── 07_baselines_and_robustness.ipynb   # cleanlab, bootstrap CIs, target variance, temperature
│   ├── 08_conditions_sweep.ipynb           # label-space size + belief-sharpness conditions
│   ├── 09_routing_operational.ipynb        # budget-recall curves, percentile sensitivity
│   ├── 10_training_dynamics_ensemble.ipynb # degradation curve, AUM/Data-Maps, ensemble  [GPU]
│   ├── 11_cifar_n_generalization.ipynb     # (optional) second real-noise domain, CIFAR-N  [GPU]
│   └── validation/
│       ├── v01_identity_and_label_invariance.ipynb
│       ├── v02_signature_table.ipynb
│       └── v03_nontriviality_rq2.ipynb     # controlled synthetic checks (no data, no GPU)
├── results/{figures,tables}/          # grayscale PNG+PDF (600 dpi) and CSV outputs
└── data/                              # download target (gitignored)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
# install the CUDA build of PyTorch matching your driver first (see pytorch.org), then:
pip install -r requirements.txt
git lfs install                                        # for the GitHub data route
```

## Reproduce

Run notebooks in order; `01`, `10`, and `11` need a GPU, the rest read `01`'s saved artifacts.

1. **`00`** downloads AlleNoise (tab-separated `offer_id · text · clean_category_id ·
   noisy_category_id`) and writes `data/allenoise_norm.parquet`; column detection is automatic.
2. **`01`** fine-tunes the classifier with **k-fold cross-fitting** and writes out-of-sample
   predicted distributions to `data/pi_memmap.npy` (float16, N×K) and `data/item_meta.parquet`.
   Recommended: `EPOCHS=10, BATCH=128, LR=4e-5, N_FOLDS=3`. On Windows set `num_workers=0`.
3. **`02`–`10`** read `01`'s artifacts (only `10` retrains) and reproduce every table and figure
   in the paper. **`07`** requires `pip install cleanlab`.
4. `notebooks/validation/` runs standalone (no data, no GPU).

The metric is a small dependency-free module:
```python
from guard_core import axis_A, axis_B, guard_by_group
W, D, H, pbar = axis_A(P)             # P: (n, K) predictions
C, kappa, q, pbar = axis_B(P, a, K)   # a: assigned labels
rows = guard_by_group(P, a, group_ids, K)
```

## Data availability

AlleNoise is a public benchmark (search "AlleNoise large-scale text classification real-world
label noise"); `00_download_data.ipynb` fetches it. It is distributed under its own license; see
its source. Large artifacts (`data/`, `*.npy`, `*.parquet`) are gitignored and regenerated by the
notebooks.

## Anonymity note

This repository is anonymized for double-blind review: it contains no authors, affiliations,
acknowledgments, or identifying links, and citations to the authors' own prior work (if any) are
written in the third person. Please keep it so until the review process concludes.

## License

For the review period, released for the sole purpose of reproducibility. A permissive license
(e.g., MIT) will be attached in the de-anonymized version.
