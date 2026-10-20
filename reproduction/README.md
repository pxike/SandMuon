# Historical source archive

The files in this directory preserve the optimizer and its local ODL kernel dependency closure from the 103,176,192-parameter FineWeb run. The `odl/optim/diagonal_weighted_pe.py` byte-for-byte source is the file whose SHA-256 appears in `manifest.json`. The production optimizer imports the packed-prefix compensated S8 path at runtime; the archived kernel modules preserve that path and its imports. `paper/s8/data/completed/s8_manifest.json` contains the recipe coefficients referenced by the source implementation.

This is provenance and source preservation, not a self-contained launcher. The original runtime also used the broader ODL model/training code and CUDA/Triton environment. The portable package in the repository root is the supported standalone API and explicitly uses eager FP32 S8 products.
