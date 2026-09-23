# Group-Level Uncertainty under Real Label Noise

*A cautionary, reproducible study on whether group-level uncertainty helps localize
real-world label noise — evaluated on the AlleNoise benchmark.*

This repository investigates a natural idea: when individual prediction confidence fails to
flag mislabeled items, can **group-level** (entity / taxonomy) uncertainty do better? We
implement **GUARD** (Group Uncertainty And Residual Decomposition), a two-axis group metric,
validate it on controlled synthetic data, and stress-test it on real e-commerce label noise.

**Headline result (honest):** on real AlleNoise noise, group-level claim–belief divergence
*does not beat* a trivial per-item aggregation baseline. The value of this repo is the
**phenomenon it quantifies**, the **failure it diagnoses**, and the **conditions it delineates** —
not a new state-of-the-art detector.

---

## Key findings

1. **Individual confidence collapses under semantic noise.** Detecting real misregistrations
   (noisy ≠ clean) from individual signals is near chance — AUROC ≈ 0.53 — and it gets *worse*
   as the classifier is trained better (a stronger model confidently memorizes plausible wrong
   labels).

2. **Naive group-level `C = JS(claim ‖ belief)` saturates in a large label space.** With ~5.7k
   classes and a diffuse classifier, `C` is dominated by belief spread outside the subtree
   (mean ≈ 0.93–1.0) and carries no signal. A principled **subtree projection** removes the
   saturation (mean ≈ 0.09).

3. **Even after the fix, GUARD does not beat trivial aggregation.** Ranking entity groups by
   true noise rate, all scores land in a weak band (Spearman 0.12–0.21, AUROC 0.58–0.62), and
   the best is a **baseline** (the 90th-percentile of per-item mismatch), statistically tied with
   the best GUARD term.

4. **The metric is mathematically sound but does not transfer.** In controlled synthetic data
   the axis-A identity `H(p̄)=W+D` holds exactly, and κ separates structured vs diffuse
   contamination perfectly (AUROC 1.0) where aggregation is at chance. The gain vanishes on real
   data because it requires a small label space and a sharp classifier.

### Numbers at a glance

| Aspect | Value |
|---|---|
| Dataset | AlleNoise — 502,310 items · 5,691 categories · 14.75% real noise |
| Classifier | HerBERT, 5-fold cross-fitted, held-out top-1 ≈ 0.553 (clean) |
| Individual detection AUROC | 1−p 0.526 · margin 0.512 · entropy 0.535 |
| Group ranking (DEPTH 4, 475 groups) | best baseline p90 → Spearman +0.210 / AUROC 0.622 |
| Best GUARD term | C_proj·κ → Spearman +0.195 / AUROC 0.623 (tied, no gain) |
| Synthetic sanity | identity error < 2e-16 · RQ2 κ AUROC 1.0 vs aggregation 0.5 |

---

## Repository structure

```
.
├── requirements.txt
├── src/
│   └── guard_core.py                 # the GUARD metric (numpy-only, importable)
├── notebooks/
│   ├── 00_download_data.ipynb              # download + profile AlleNoise (GitHub-LFS / Zenodo)
│   ├── 01_finetune_and_predict.ipynb       # HerBERT k-fold -> out-of-sample p_i  [GPU]
│   ├── 02_layer1_detection.ipynb           # individual misregistration detection
│   ├── 03_layer2_signature.ipynb           # group construction + noise anchor
│   ├── 04_nontriviality_rq2.ipynb          # structured vs diffuse separability
│   ├── 05_routing_policy.ipynb             # review-budget routing
│   ├── 06_taxonomy_group_analysis.ipynb    # taxonomy projection + decisive comparison (no rerun)
│   └── validation/                         # controlled synthetic de-risking
│       ├── v01_identity_and_label_invariance.ipynb
│       ├── v02_signature_table.ipynb
│       └── v03_nontriviality_rq2.ipynb
├── results/
│   ├── figures/                      # grayscale PNG + PDF, 600 dpi
│   └── tables/                       # CSV outputs
└── data/                             # download target (gitignored)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
# install the CUDA build of PyTorch that matches your driver first:
pip install torch --index-url https://download.pytorch.org/whl/cu121   # example
pip install -r requirements.txt
git lfs install                                         # for the GitHub download route
```

## Reproduce

Run the notebooks in order; only `01` needs a GPU.

1. **`00`** downloads AlleNoise and writes `data/allenoise_norm.parquet`.
   Column auto-detection handles the tab-separated schema
   (`offer_id · text · clean_category_id · noisy_category_id`).
2. **`01`** fine-tunes HerBERT with **k-fold cross-fitting** and writes out-of-sample predicted
   distributions to `data/pi_memmap.npy` (float16, N×K ≈ 5–6 GB) plus `data/item_meta.parquet`.
   Recommended: `EPOCHS=10, BATCH=128, LR=4e-5, N_FOLDS=3` (converges to clean top-1 ≈ 0.55).
   On Windows set `num_workers=0` in the DataLoaders.
3. **`02`–`06`** are fast and read only `01`'s artifacts (no retraining). `06` reproduces the
   taxonomy projection and the decisive GUARD-vs-baseline comparison, and regenerates all figures.

Controlled synthetic checks in `notebooks/validation/` run standalone (no data, no GPU).

---

## The metric (`src/guard_core.py`)

For a group `g` of items with predicted distributions `P` and assigned labels `a`:

- **Axis A — homogeneity (label-free):** `H(p̄) = W + D` (BALD / Jensen-gap identity), where
  `W` is mean per-item entropy (aleatoric) and `D ≥ 0` is between-member disagreement (epistemic).
- **Axis B — contamination (label-aware):** `C = JS(q_g ‖ p̄)` (claim–belief divergence) and
  `κ = 1 − H(r̃)/log K` (residual direction concentration), with the **subtree-projected**
  variant `C_proj` recommended for large label spaces.

```python
from guard_core import axis_A, axis_B, guard_by_group
W, D, H, pbar = axis_A(P)
C, kappa, q, pbar = axis_B(P, a, K)
rows = guard_by_group(P, a, group_ids, K)
```

---

## Limitations

- No native seller IDs in AlleNoise, so entity groups are taxonomy subtrees (a proxy).
- Single classifier (HerBERT, top-1 ≈ 0.55) and single platform; a stronger model could shift
  the quantitative picture.
- Findings are specific to semantically-disguised, instance-dependent noise; they need not hold
  for synthetic class-conditional noise.

## Data & citation

Data: **AlleNoise** (Rączkowska et al., AISTATS 2025), arXiv:2407.10992,
repo `allegro/AlleNoise`. Please cite the dataset when using it.

```bibtex
@misc{group_uncertainty_label_noise,
  title  = {Group-Level Uncertainty under Real Label Noise: A Cautionary Study on AlleNoise},
  author = {<your name>},
  year   = {2026},
  note   = {https://github.com/<user>/group-uncertainty-label-noise}
}
```

## License

MIT (suggested) — add a `LICENSE` file. AlleNoise data is under its own license; see the source.
