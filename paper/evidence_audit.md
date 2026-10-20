# SandMuon evidence audit

This audit distinguishes the completed full-recipe FineWeb run from the smaller exploratory screens and application demonstrations. The bundle contains manifests, scalar summaries, validation/update traces, and plots; it contains no model weights.

## Main FineWeb runs

The four traces use the same 103,176,192-parameter GPT-2 configuration (9 layers, width 768, 12 heads, context 1,024), FineWeb-Edu GPT-2 token shards at revision `a33f75d78c7f74236fb03754ec1eb3cc77507d64`, seed 1337, effective batch 128, microbatch 16, LR 0.01, and FP32 weights/state with BF16 forward/backward and matrix products. Loss is validation cross-entropy in nats per GPT-2 token.

| Recipe | Final validation loss | Updates (planned) | Tokens | Training seconds | Matrix momentum |
|---|---:|---:|---:|---:|---:|
| Raw Muon | 3.310583 | 16,000 (16,000) | 2.097B | 36,890.38 | 0.95 |
| Polar Express (PE) | 3.304063 | 16,000 (16,000) | 2.097B | 37,807.67 | 0.95 |
| S8 | 3.301541 | 16,000 (16,000) | 2.097B | 40,051.24 | 0.95 |
| SandMuon rank-one | 3.277938 | 15,106 (16,000 cap) | 1.980B | 37,808.94 | 0.8 |

The SandMuon record is the complete time-budget run. It stopped at the PE reference's measured training duration, not at the 16,000-update cap. Its recorded final validation loss is lower than the archived PE final loss, but the rows are complete recipes across separate runs and hosts. Treat this as descriptive evidence only: it does not establish a matched optimizer win, a causal geometry effect, or a strict speedup.

Material protocol differences include:

- SandMuon uses A as an EMA-0.97 diagonal centered-activation variance plus activation-mean outer product, applied by exact rank-one whitening. B is an EMA-0.97 full-effective-batch relative real-cotangent square shape. The archived references do not use this geometry.
- SandMuon uses compensated packed-prefix S8; PE uses the published five-stage Polar Express polynomial; raw Muon uses its raw polarizer; S8 uses the fitted five-stage S8 recipe.
- SandMuon uses a wall-clock cosine schedule and completed 15,106 updates. The archived references use fixed 16,000-update cosine schedules.
- PE warms up for 100 updates. SandMuon uses the same 100/16,000 warmup fraction in its wall-clock schedule, about 236.3 seconds of the PE reference budget; this does not mean 100 literal SandMuon updates.
- SandMuon matrix momentum is 0.8; archived Muon, PE, and S8 use 0.95. The auxiliary AdamW also differs: SandMuon records LR ratio 0.1, betas (0.9, 0.999), and weight decay 0.01; the S8 paper protocol records LR ratio 0.1, betas (0.9, 0.95), and weight decay 0.
- The reference runs share model, seed, data revision, batch, LR, and nominal GPU class. Host identity and exact same-host provenance are not retained for the full set, so timing comparisons are not controlled.

The time-aligned PE validation record nearest to SandMuon's final training time is 3.304063 at 37,807.67 seconds (step 16,000); SandMuon finished at 37,808.94 seconds with loss 3.277938. This proximity in elapsed time does not remove the schedule, momentum, auxiliary-optimizer, and host-provenance differences.

The SAND run also had a time-validation guard: every 2,000 updates, it compared validation loss with the PE trace linearly interpolated at equal training time, with patience one. The guard did not trigger, and the run ended at its wall-clock budget. Because these validation checkpoints were observed by run control, they are exploratory observations rather than a blind test set; the final validation result should be interpreted accordingly.

## Provenance and integrity

- The SAND manifest's SHA-256 for `odl/optim/diagonal_weighted_pe.py` matches the current file (`d74e7c93…93452d0`), and its `odl/ops/dense_s8_compensated_packed_prefix.py` hash matches (`299aaa14…deaf05c`).
- The SAND manifest's recorded `run_muon_100m.py` hash differs from the current file. Therefore the current runner is not evidence of an exact code reconstruction; the preserved manifest and traces are the run record.
- The saved PE-baseline manifest, budget, and validation hashes match the SAND manifest's recorded baseline hashes. The archived S8 completion receipt marks 16,000 steps complete, and its recorded source checks pass.
- The centered-screen source identity hashes in the selected arm manifest all match the corresponding current local files; the per-file checks are included in `data/summary.json`.
- Public copies omit Vast instance identifiers. The archived manifests identify RTX 3090-class hardware but do not provide a complete host mapping. The bundle retains no checkpoints or weights.

## Centered-statistic screen

The completed ten-arm screen used a 4-layer, width-320 GPT-2 with 21,055,680 total parameters, including 4,915,200 SAND matrix parameters, context 128, FineWeb-Edu GPT-2 tokens at the same pinned revision, seed 1337, and 1,000 updates per arm. It used eager FP32 polynomial arithmetic and is explicitly different from the compensated BF16 production run. The comparison guard accepted the arms; the intentional variants changed centering, A/B statistic rates, B shrinkage/exponent, or direction handling. The selected scalar results are:

| Arm | Loss after 1,000 updates |
|---|---:|
| Baseline | 5.871900 |
| Diagonal A | 5.864832 |
| Tempered B | 5.867986 |
| Aligned | 5.883294 |

This is one-seed screening evidence, not a replicated ablation. `centered_screen_ablation.pdf` plots these four selected arms as points on an explicitly zoomed loss axis; the complete ten-arm results and manifests are preserved under `data/centered_screen/`.

## Application runs

Two complete, single-optimizer aesthetic-generation runs are included as feasibility demonstrations. The pixel generator had 12,768,259 parameters and completed 10,000 updates in 16,768 seconds; it resumed from a saved checkpoint, reached best adjusted reward 1.8601, and ended at 1.1855. The transformer generator had 12,570,048 parameters and completed 10,000 updates from random initialization in 30,899.9 seconds; best adjusted reward was 2.7186 and final adjusted reward 2.5312. Both use the S8 polynomial in FP16 CUDA GEMMs on a Turing/T4 fallback path, which differs from the BF16 FineWeb production configuration. Neither run includes optimizer controls, so neither supports a comparative optimizer claim. An incomplete pixel-query continuation is excluded.

## Figures and machine-readable files

- `figures/loss_vs_tokens.pdf` plots the four FineWeb validation traces against consumed tokens, with a full-range panel and a late-training zoom beginning at 1.3B tokens. The zoom uses only recorded points: SandMuon ends at 1.980B tokens and the three references end at 2.097B; no line is extended beyond any run's last validation record. It is descriptive and does not imply a matched comparison.
- `figures/centered_screen_ablation.pdf` shows the selected four-arm screen.
- `data/summary.json` and `data/main_results.csv` contain the extracted full-run rows and protocol caveats. Original copied manifests, budgets, and traces are in `data/source_runs/`; screen and application records are in their corresponding subdirectories.
- `python paper/sandmuon/tools/build_evidence.py` rebuilds tables and plots from local archived sources; `python paper/sandmuon/tools/build_evidence.py --plots-only` rebuilds plots using only the files in this bundle, so it remains usable when the paper directory is copied without ODL archives. Neither mode trains or evaluates a model.
