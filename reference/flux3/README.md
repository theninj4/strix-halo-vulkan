# reference/flux3 — FLUX 3 Image Klein, copied verbatim 2026-10-15

Not imported by anything; read and diffed against (research/flux3-vertical.md X0).

- `bfl/` — black-forest-labs/flux-3-image-klein `src/flux3/` at `45e57c9fb06f9751cdced29a65c5d2171003be05`
  (2026-10-15), plus its `LICENSE`. **The spec**: semantics (reference preparation, cropping, the sampler)
  are settled here.
- `diffusers/` — huggingface/diffusers `main` at `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941` (2026-10-15):
  `models/transformers/transformer_flux3.py` and `modular_pipelines/flux3/*.py` (in `modular/`).
  **The oracle**: every `reference/dump_flux3_*.py` runs this code, installed in `.venv-flux3`.
