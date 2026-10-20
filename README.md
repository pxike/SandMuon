# SandMuon

SandMuon is a standalone PyTorch package for the diagonal sensitivity-weighted Muon optimizer used in the 103,176,192-parameter FineWeb run. The production variant is `RankOneAWeightedPE`; `DiagonalWeightedPE` exposes its diagonal activation-only counterpart. The portable API selects trainable `torch.nn.Linear` weights by default, so it works with ordinary PyTorch models. This is a generic adaptation of the selected parameter layout: the historical GPT run selected only `transformer.h.*` linear weights and treated all remaining parameters as AdamW auxiliaries. Pass `module_filter(name, module)` to reproduce that layout in a GPT-style model. The portable API shares activation statistics only for the exact historical `transformer.h.<layer>.attn.{q,k,v}_proj` naming convention.

The original GPU implementation and its kernel dependency files are preserved verbatim under [`reproduction/`](reproduction/). The portable API is an eager FP32 reference adaptation. It keeps the optimizer statistics, update order, coefficients, whitening and grafting rules, while replacing the CUDA BF16 compensated kernels with ordinary PyTorch matrix products. Its update numerics and speed are therefore different from the historical GPU run. No performance comparison is implied.

## Method

For each selected linear layer, the optimizer accumulates activation second moments and real-loss output-cotangent squares across every microbatch before one optimizer step. The activation statistic is `E[x²]`; in the rank-one variant it is represented as `diag(var(x)) + mean(x) mean(x)ᵀ`, with an analytic rank-one whitening action. The cotangent statistic is the relative output-channel shape of `E[δ²]`, normalized to mean one. Both statistics use an EMA of 0.97 per optimizer update. The portable diagonal factor clamps normalized values at `1e-6`, which keeps zero-signal inputs finite; the historical kernel path is preserved separately. Split `q_proj`, `k_proj`, and `v_proj` layers share activation statistics when their names and matching inputs follow the production naming convention.

The optimizer applies a diagonal or rank-one left and right whitener to the Nesterov Muon source, evaluates the fixed five-stage S8 polynomial in the whitened coordinates, then maps the result back. The production aspect multiplier is applied before norm matching and cancels there; the final direction has Frobenius norm `sqrt(output_rows)`. Matrix momentum is 0.8. Other trainable parameters use AdamW at 0.1 times the supplied learning rate, with betas `(0.9, 0.999)` and weight decay `0.01`.

## Install and use

Install PyTorch for your platform, then install this package from the repository root:

```bash
python -m pip install .
```

Use one optimizer step after accumulating the complete effective-batch gradient. The forward/backward hooks collect statistics over each microbatch automatically.

```python
import torch
from sandmuon import RankOneAWeightedPE

model = MyModel()
optimizer = RankOneAWeightedPE(
    model, lr=1e-2,
    module_filter=lambda name, module: name.startswith("transformer.h.")
        and isinstance(module, torch.nn.Linear),
)

for microbatch in effective_batch:
    loss = loss_fn(model(microbatch.inputs), microbatch.targets)
    (loss / number_of_microbatches).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

# Remove hooks when training ends.
optimizer.close()
```

Run the model forward and backward with gradient tracking enabled, as in ordinary PyTorch training. The optimizer state is compatible with PyTorch `state_dict()` and `load_state_dict()` for save/resume. Use `DiagonalWeightedPE(model, lr=...)` for diagonal activation scaling without the activation-mean rank-one correction. Do not call `step()` until all microbatches belonging to the effective batch have completed backward; the optimizer checks that activation and cotangent row counts match.

## Historical 103M run

The archived `reproduction/manifest.json` records the exact run configuration, source hashes and dependencies. It pins GPT-2 vocabulary 50,257, context 1,024, 9 layers, width 768, 12 heads, batch 128, microbatch 16, a 16,000-update cap and 37,807.672345-second training budget (the run stopped at 15,106 updates), seed 1337, learning rate 0.01, and FineWeb revision `a33f75d78c7f74236fb03754ec1eb3cc77507d64`. The recorded precision was FP32 weights and optimizer state, with BF16 autocast forward/backward and BF16 S8 products. The run completed 1,979,973,632 training tokens with final validation loss 3.2779383659. The complete-recipe PE reference reports final validation loss 3.304063; this whole-recipe comparison does not isolate optimizer effects. The source archive is the implementation used in the run; this package's portable eager FP32 path is provided for reuse and CPU validation, not bitwise reproduction of that BF16 CUDA execution.

## Tests

Run the CPU reference tests with limited thread use:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests compare the rank-one whitener and mapback to dense matrix oracles, check microbatch accumulation, optimizer state resume, orientation handling, and finite behavior for zero signal.

## Citation

See [`CITATION.cff`](CITATION.cff). The preliminary manuscript, figures, and compact evidence bundle are in [`paper/`](paper/). See [`paper/main.pdf`](paper/main.pdf) for the paper and [`paper/evidence_audit.md`](paper/evidence_audit.md) for the comparison audit. The public repository is https://github.com/pxike/SandMuon. No DOI or preprint identifier is claimed.
