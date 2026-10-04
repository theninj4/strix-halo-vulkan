# CLASSIFICATION — typed decisions with calibrated probabilities (Kev)

> **Frozen 2026-10-04: Kev is retired.** Rune v3 (Gemma 4 26B-A4B,
> surogate's decisions v1) replaced it and answers `/v1/systemone` through a
> translation; the live plan is [`rune-vertical.md`](rune-vertical.md). Kev's
> code (`kev/`, `cmd/kev`, `backend/kev.go`, its shaders and its oracle) is
> in git history before that date; `cmd/kevload`, `reference/kev_fixtures.json`,
> `models/kev-suites/` and `kev_gemm_q8_glu.comp` (Rune's int8 GEMMs) stay.

> **Moved to `research/` on 2026-10-02** — this was `CLASSIFICATION.md` at the repo
> root, and code comments citing `CLASSIFICATION.md` resolve here. The vertical is
> **not closed**:
> § Handoff below is where a session resumes, and it carries the open items. The long write-ups (K7.1–K7.7,
> K8, K9) are broken out into `k*.md` files beside this one, linked where each
> stage's item was; K-stages and decisions D1–D5 still resolve here.

> **Live tracking and session handoff doc, opened 2026-09-25.** Stage letters
> are **K**. When the vertical closes, this file is frozen in place and [`README.md`](README.md)'s index marks it closed. Until then: tick a stage when its gate passes, put
> its measured numbers under it, and keep **§ Handoff** at the bottom
> current. A new session should be able to start from there.

## What we are building

`GOALS.md` item 6: classification via `kev-4b`. The product is TypeSafe's
**System One** contract, `POST /v1/systemone`. One text (the *state*) and any
number of typed questions go in. A probability distribution per question
comes out of **one prefill, with no text generated**. The TypeSafe Python SDK
must work against our server unchanged, as it does against Kev's own.

### Jev, and why "a JSON classifier" is not the point

Jev is TypeSafe's hosted, closed decision model (launched Sep 2026). The only
architectural source is Archer Hume's black-box study,
[*Jev's Architecture Unmasked*](https://archerhume.com/posts/jevs-architecture-unmasked)
(17 Sep 2026, 10,000 API calls against `jev-1.13.0`). What it establishes, in
order of confidence:

1. **A readout, not a decode loop.** TypeSafe says so: "outputs all
   probabilities in parallel instead of autoregressively generating by
   token". `usage.output_tokens` is a **billing figure** computed from the
   serialised response. It changes with question ids the model never sees,
   and latency does not track it (200 options returned as fast as 2).
2. **Shared state, isolated questions.** Token accounting is exactly
   additive. A secret placed in a *sibling question* is invisible (p = 0.00),
   and the same secret in the *state* is visible (0.90). Server time is flat
   to ~100 questions and grows slowly after. Limits: ~32k tokens a branch
   (state plus one question), ~64k a request with the state counted once.
   That is a prefix cache with independent causal suffixes (Hydragen, DeFT).
3. **A causal backbone.** It is inferred, not measured.
4. **Options interact before the choice.** Appending an irrelevant fifth
   option moves the log-odds between two existing ones in all ten blocks
   (mean −0.28). So it is not independent per-option logits plus a softmax.
   A reference card placed *last* in the option list is read correctly 16/16
   times; placed first, 12/16. The readout sees the whole ordered list.
   Two designs fit: a final-position slot head, or a **pointer** scorer.
   Fake delimiters injected in option text never displace real options, so
   the boundaries are unforgeable. At most 255 options (2⁸−1).
5. **Trained on outcomes (RLCD), then `confidence` is plain arithmetic.**
   For a choice with K > 1 options, `(p_max − 1/K)/(1 − 1/K)`. It is not a
   learned estimate of being right.
6. Sparse MoE backbone (least certain; nothing else depends on it).
7. Branches are scheduled as a batch; answers are not bit-deterministic.

### Kev: the open implementation we build

[Kev](https://github.com/jaredpalmer/kev) (Apache-2.0, Jared Palmer; read at
`9fdf054`, 2026-09-25) is that design, trained and published.
[`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) at
`139fdd94` (round 10, 2026-09-24) is **three things on
`Qwen/Qwen3.5-4B-Base@1001bb4d`**:

- a **rank-16 LoRA** (α 32, so scale 2.0) on every projection: q/k/v/o,
  gate/up/down, and the DeltaNet `in_proj_qkv/z/a/b` and `out_proj`. 496 fp32
  tensors, 33.8 M parameters, **merged into the base at load** (`W += 2·B·A`
  in fp32, rounded once, which is what Kev's server does);
- a **pointer head**: two `Linear(2560 → 256)` with bias, `q` applied to the
  `<decide>` hidden state and `k` to each option's `</opt>` hidden state.
  `z_i = (k(h_opt_i) · q(h_decide)) / sqrt(256) / T`, then softmax over the
  question's options;
- a **temperature**, `T = 2.406050072164233`, fitted on held-out rows. It
  never changes the argmax. `head.pt` is a torch pickle, so K1 converts it
  once to safetensors plus JSON.

Published accuracy for Kev-4B: 0.817 / 0.838 on new sources (dev / locked
test), against Jev's 0.857 on dev. Kev's GPU reference numbers, model time
per request: **H100 18.1 ms** (six questions, short text) and 89.4 ms (five
questions over a 2,200-token text; 22.5 ms when the text is cached). An
L40S takes 41.5 / 145.2 ms, and an Apple M5 721 ms (five questions).

### The details at the start that decide everything

These are the places where a reasonable reading gives a plausible, wrong
model. Each one is a K2 or K4 gate.

1. **The question type is `"noul"`**, not `"bool"`, in requests and
   responses alike. The answer field is also `noul` (p(yes)). Choice and
   score are `choice` and `score`.
2. **There is no BOS and no chat template.** The row is the delimiters and
   the user text, nothing else. The delimiters are reused Qwen tokens with
   **non-obvious roles**:

   | role | token | id | notes |
   |---|---|---|---|
   | `<state>` | `<\|fim_prefix\|>` | 248060 | first token of every row |
   | `<q>` | `<\|fim_middle\|>` | 248061 | starts a question |
   | `<opt>` | `<\|box_start\|>` | 248049 | starts an option |
   | `</opt>` | `<\|box_end\|>` | 248050 | **the option's readout position** |
   | `<decide>` | `<\|fim_suffix\|>` | 248062 | **the question's readout position**, last token |

   `<decide>` is `fim_suffix` and `<q>` is `fim_middle`. The ids come from
   `tokenizer.json`, not from this table, and a test asserts them.
3. **User text cannot forge a delimiter.** Before tokenising any state,
   instruction or option text, every `<|name|>` (`[A-Za-z0-9_]+`) is rewritten
   to `<¦name¦>` (U+00A6). Then the text is tokenised with
   `add_special_tokens=False`. The fim tokens are *not* `special` in
   `tokenizer.json`, so HF would match them in raw text; this rewrite is what
   stops it.
4. **The row layout** for question k is
   `[<state>] state… [<q>] instr… ([<opt>] option… [</opt>])×K [<decide>]`.
   Positions are contiguous from 0 through the state and then through the
   branch, so every question's branch starts at position `Ls`.
5. **The rendered text is Kev's `api.render`, byte for byte.** A string is
   itself. `None` is empty. A number or bool is Python's `str()`, so
   `True`, `1.5`. A list becomes `- item` lines. A dict becomes `key: value`
   lines, with nested values on the following lines indented two spaces per
   level. Instructions may be empty, which gives just `<q>`.
6. **Option text.** A choice option is `name` if its description is null or
   empty, and `name: <render(desc)>` otherwise. A **noul** is two options in
   the order **`[no, yes]`**, each optionally `no: <criteria.false>` or
   `yes: <criteria.true>`, and the answer is `p[1]`. A score's options are the
   rendered levels in order. The answer is the expected index Σ i·pᵢ, with
   `legend` and `probabilities` keyed by index strings.
7. **Readout on the final-normed hidden state**, in fp32, and only at the
   `</opt>` and `<decide>` positions.
8. **Limits (Kev's serving limits).** The state is truncated silently to
   8,191 tokens plus the marker (`strict=False`), and a row (state plus
   branch) over 8,192 tokens is a **422**. Options: 1–255. Training only saw
   states of up to 384 tokens, so longer ones work, but less well.
9. **Confidence.** A choice with K = 1 has confidence 1, otherwise
   `(p_max − 1/K)/(1 − 1/K)`. Score: `1 − Σ pᵢ·|i − mode| / (L − 1)`,
   which is Kev's approximation, since TypeSafe's formula is unpublished.
   Probabilities are rounded to 4 decimals. `usage.input_tokens` is the
   packed token count (state once, plus every branch). `usage.output_tokens`
   is the token count of `json.dumps(answers)`.

### The backbone: Qwen3.5-4B-Base (text only)

32 layers of hidden 2560, in a pattern of three **Gated DeltaNet** layers
and then one **gated full-attention** layer (so 24 + 8). The MLP is dense
SwiGLU at 9216. Vocab 248 320, embeddings tied (we never need the LM head).
The vision tower and the MTP layer are not loaded. Against things this repo
already runs:

| | Qwen3.5-4B (this) | qwen3.8-flash-next (`llm/`) | Qwen3-4B (`zimage/qwen`) |
|---|---|---|---|
| mixer | GDN ×24 + attention ×8 | GDN ×36 + attention ×12 | attention ×36 |
| GDN heads | 16 k × 128, **32 v** × 128, conv 4 | 16 k, 48 v | – |
| GDN V-head order | **HF grouped**: v-head h reads k-head `h / 2` | GGUF tiled (`h % 16`) | – |
| GDN output gate | RMSNorm(o)·w · **silu(z)** | · sigmoid(z) | – |
| attention | 16 q × **256**, 4 kv, q-proj emits q+gate, sigmoid gate | 24 q × 256, + QSA | 32 q × 128 |
| rope | NeoX on the **first 64 of 256** dims, θ = 10⁷ (M-RoPE collapses on text) | same | full, θ = 10⁶ |
| norms | **zero-centred**: `x·rsqrt(ms+ε)·(1+w)`, q/k norm per head, ε 1e-6 | (+1 folded by the converter) | `·w` |
| FFN | dense SwiGLU 9216 | MoE + HC + PLE | dense 9728 |

GDN per token and v-head (HF `torch_recurrent_gated_delta_rule`): q and k
are l2-normalised with `rsqrt(Σx² + 1e-6)` and q is scaled by 1/√128.
`g = −exp(A_log)·softplus(a + dt_bias)` and `β = sigmoid(b)`. Then
`S ← S·eᵍ`, `S ← S + k⊗β(v − kᵀS)`, `o = qᵀS`. Before all that,
`in_proj_qkv` goes through a depthwise causal conv (kernel 4, no bias) and
a SiLU. The conv history and S are what a row inherits from the state.

Weights (fp16 on the device): **7.13 GB**. That is 24 × 112.8 M (GDN plus
MLP) and 8 × 107.5 M (attention plus MLP). The embedding table is 248 320 × 2560
= 636 M values, 1.27 GB fp16, kept on the host, with rows gathered per token.

## How a request runs (the design)

Kev's reference serves the hybrid backbone in its **row form**, because
recurrent layers cannot honour a block-causal mask. It encodes the state once
(KV plus every GDN layer's S and conv tail), then runs each question as a
causal row continuing from a copy of that cache. It computes the same
function as the packed block-causal form (`tests/test_model.py`, 4e-6).

We do it as **one segmented pass**: the state segment followed by every
question's branch, laid end to end the way Kev's packed encoding lays them
out (`encode()`'s `ids/seg/pos`):

- **Projections, norms and MLP** see the packed tokens as one M dimension.
  With Q questions of ~40 tokens that is a single GEMM of `Ls + ΣLq` rows,
  not Q+1 small ones. This is where the batching pays.
- **GDN** runs two scans per layer. The first is the state segment from
  zero. The second is every branch *in parallel*, each starting from the
  state's final S and conv tail, which it reads and does not copy. Branch
  final states are discarded, so the scratch is per layer, not per row.
- **Attention** is Kev's block-causal mask exactly. A branch token sees the
  state's keys plus its own branch's earlier keys, and no other branch's.
  The state's KV is written once.
- **The readout** reads back only the `<decide>` and `</opt>` rows
  (ReadRow; the arena reads at 0.2 GB/s), then runs the pointer head on the
  host. That is Q + ΣK dot products of 256.

A **prefix cache** (K7) keeps the state segment's KV, S and conv tails,
keyed by its token ids. A repeated state then runs its branches only, which
is Kev's "same text again" column. Requests are **batched** across callers
(K7) by concatenating their segments, since nothing crosses a segment.

Precision: fp16 weights and GEMM operands with fp32 accumulation. The
residual stream, GDN state and scan are fp32. Kev serves bf16 by default and
publishes fp32 numbers, and its bf16 path is within ~0.03 of fp32 with one
argmax flip in 300. Our gate is against its **fp32** path.

## Budget

- Device: 7.13 GB of weights, plus activations sized for the
  `-kev-tokens` packed budget (default 16,384, Kev's `SERVE_MAX_PACKED`).
  The KV of an 8k state is 8 layers × 8192 × 4 × 256 × 2 × 2 B = 268 MB. GDN
  scratch is 32 v-heads × 128² × 4 B = 2 MB per branch in flight, per layer,
  reused.
- Host: the embedding table at 1.27 GB fp16.
- Deployment: the **small-verticals machine** (the LLM has the other one to
  itself). About 9 GB beside image (~32 GB) and the rest (~3 GB).

## Stages

- [x] **K0 — ground truth and decisions** (2026-09-25). Above. Sources: the
  essay, Kev's `README.md`, `kev/model.py`, `kev/api.py`, `kev/serve.py`,
  `kev/checkpoint.py`, HF `modeling_qwen3_5.py` (transformers 5.17.0), the
  kev-4b and Qwen3.5-4B-Base configs. **Decisions:**
  - **D1.** Implement Kev-4B, not a Jev guess. The API is TypeSafe's, as Kev
    serves it.
  - **D2.** A new package `kev/`, with its own graph over the four-arena
    layout and push block of `shaders/dit_common.glsl`, so that it reuses
    `dit_gemm`'s fp16 WMMA ladder, the RMS-norm pack and SwiGLU unchanged.
    **Not** `llm.DeltaNetGPU`/`AttnGPU`: they carry GGUF head order, the
    sigmoid gate, Q8 banks, QSA and slots. Bending them would put the LLM's
    hot path at risk for a 4B model. Their GLSL is the starting point for
    the new scan and attention kernels.
  - **D3.** No Go CPU model. The oracle is Kev's own code running in fp32
    under transformers 5.17.0 (with peft 0.21.0 added to `.venv`), dumped
    per layer. Kernels are unit-tested against small CPU loops in tests.
  - **D4.** One segmented pass (above) rather than Kev's copy-the-cache
    rows. It is the same function, and the gate is Kev's probabilities.
  - **D5.** Serve on the existing `cmd/serve` behind `-kev`, as
    `POST /v1/systemone`. `/v1/models` stays OpenAI-shaped and lists
    `kev-latest` and `jev-latest`. TypeSafe's `{"models":[…]}` card is not
    served, since the SDK's `system_one` does not read it.

- [x] **K1 — weights and the oracle** (2026-09-25). `models/Qwen3.5-4B-Base`
  is at `1001bb4d` and `models/kev-4b` at `139fdd94`.
  `reference/convert_kev_head.py` writes `head.safetensors` plus `head.json`
  (T = 2.406050072164233, head_dim 256). `reference/kev/` vendors Kev's
  `model.py`, `api.py` and `checkpoint.py` at `9fdf054`, unchanged.
  `reference/dump_kev.py` loads through Kev's own `Checkpoint.load` in fp32
  with the adapter merged, the base redirected to the local snapshot. For the
  four requests in `reference/kev_fixtures.json` it writes
  `reference/out/kev/<name>/{record.json, rows.safetensors}`: the record,
  the packed encoding, Kev's served probabilities, `json.dumps(answers)`,
  usage, and every row's embeddings, all 32 layer outputs and its final
  hidden state. `reference/dump_kev_tokens.py` adds 30 adversarial strings
  through `user_tokens`. **Measured:** Kev's serving path (the state once,
  then rows from its cache) and the plain rows agree to **3.3e-7**, so the
  oracle is self-consistent. Its answers for the README ticket (returns 0.509,
  escalate 0.727) are not the README's example (0.47, 0.93). That example was
  run in bf16 on an M5 and predates round 10, so the dump is the reference.

- [x] **K2 — the request layer in Go** (2026-09-25). `kev/value.go` is an
  ordered JSON value that keeps numbers as literals, plus `Render` (Kev's
  `api.render`, including Python's `str()` of ints, floats, bools and None,
  and `lstrip`'s notion of whitespace). `kev/api.go` holds request parsing
  and validation, `ToRecord`, `ToAnswers`, the confidences, 4-decimal
  rounding, `json.dumps`-exact serialisation for `output_tokens`, and the
  response. `kev/encode.go` holds the delimiter escape and the packed
  encoding. **Gate passed:** rendered text, ids, seg, pos, option markers,
  readout indices, `input_tokens`, `json.dumps` bytes and `output_tokens` are
  identical to Kev on every fixture, and the 30-string tokeniser sweep is
  identical. **Found on the way:** `zimage/tokenizer.Load` never read the
  pre-tokenizer regex from `tokenizer.json` and always ran Qwen2's. Qwen3.5's
  is the `\p{M}` variant, and on NFC text with combining marks (Hindi's vowel
  signs, say) the two tokenise differently. `Load` now reads the regex and
  refuses one it does not implement. Z-Image, Qwen3-Embedding and
  Qwen-Image all carry Qwen2's, so they are unchanged (their tests pass).
  **Known gap:** there is no NFC normalisation, as for every tokenizer in
  the repo, so the one non-NFC string in the sweep is asserted to *differ*.

- [x] **K3 — load and merge** (2026-09-25). `kev/load.go` computes
  `W + (B @ A) * 2` in fp32, in peft's order, from the base's bf16. All
  496 adapter tensors are consumed, and the loader fails if any projection
  is left unmerged. `packInto` refuses a merged weight that overflows fp16;
  none does. Staging is **10.6 s**, and residency is 7.1 GB of fp16 banks
  plus the 1.27 GB bf16 embedding table on the host. The merge is checked
  end to end by K4 rather than per projection: it is the layer outputs that
  have to agree.

- [x] **K4 — one row on the GPU** (2026-09-25). `kev/gpu.go` plus five new
  kernels (`shaders/kev_gdn_prep`, `kev_gdn_scan`, `kev_gdn_norm`,
  `kev_attn_prep`, `kev_attention`), with dit_gemm's fp16 ladder,
  `dit_norm_scale_f16`, `dit_swiglu_f16` and `dit_gate_add` reused as they
  are. **Gate passed** (`TestGPURowMatchesKev`): the residual stream after
  every layer is within **2.0e-3 relative** of Kev's fp32 (3e-4 at layer 0,
  growing slowly), and the final-normed rows within **9.8e-4**.

- [x] **K5 — the segmented pass** (2026-09-25). **Gate passed**
  (`TestGPUProbsMatchKev`): every fixture as one packed pass of all its
  questions is within **4e-4** of Kev's fp32 probabilities (max |dp| over 13
  questions and 36 options), with **no argmax flips**. Kev's own bf16 server
  is ~0.03 from fp32. Isolation (`TestGPUQuestionsAreIsolated`): rewriting a
  question or removing it leaves every other answer **bit-identical**.

- [x] **K6 — served** (2026-09-25). `api/systemone.go`
  (`SystemOneBackend`; the body crosses as bytes, see API.md),
  `backend/kev.go`, and `cmd/serve -kev -kev-model -kev-base -kev-tokens`.
  **Gate passed:** the README ticket over HTTP comes back identical across
  repeats, within 3e-4 of the oracle, with `x-typesafe-request-id` echoed.
  An unknown type, empty questions and a non-JSON body are 422s. `typesafe-sdk`
  0.6.0 from PyPI (in a scratch venv, not `.venv`) runs Kev's README client
  snippet unchanged, `Noul`, `Choice` and `Score` all parse, and a client with
  no `model` (so `jev-latest`) is answered. `api/systemone_test.go` covers
  501, 422 and the request id.

- [ ] **K7 — speed.** Measured first (`TestGPUProfile`, kernel time summed
  per label, one dispatch per submit):

  | kernel | 101 tokens, 3 q | 494 tokens, 3 q |
  |---|---|---|
  | in proj + out proj + gate + up + down (the GEMMs) | **52.3 ms (90%)** | 110.7 ms (71%) |
  | GDN prep / scan state / scan branches / norm | 4.1 ms | 24.7 ms |
  | attention prep + attention | 0.3 ms | 6.3 ms |
  | swiglu, norms, residuals | 1.8 ms | 13.7 ms |
  | total | 58.1 ms | 154.9 ms |

  Through HTTP the README ticket was **64 ms** of model time, against Kev's
  L40S at 41.5 ms and H100 at 18.1 ms. A short request is a read of the
  weights: 7.1 GB of fp16 in 52 ms is 137 GB/s, where the LLM's research
  (P4, P1b) puts this bus at ~242 GB/s and its best weight streaming at
  180–200.

  - [x] **K7.1 — the int8 bank** (2026-09-25). Broken out to [`k7.1-int8-bank.md`](k7.1-int8-bank.md).

  - [x] **K7.2 — the prefix cache, and requests longer than a pass** (2026-09-25). Broken out to [`k7.2-prefix-cache.md`](k7.2-prefix-cache.md).

  - [x] **K7.3 — the attention on the matrix cores** (2026-09-25). Broken out to [`k7.3-wmma-attention.md`](k7.3-wmma-attention.md).

  - [x] **K7.4 — the GDN scan** (2026-09-25). Broken out to [`k7.4-gdn-scan.md`](k7.4-gdn-scan.md).

  - [x] **K7.6 — the SwiGLU fusion, the GEMM grid order, and the GDN prep** (2026-09-25). Broken out to [`k7.6-glu-fusion.md`](k7.6-glu-fusion.md).

  - [x] **K7.5 — batching across requests** (2026-09-25). Broken out to [`k7.5-batching.md`](k7.5-batching.md).

  - [x] **K7.7 — the WMMA residue was stale V** (2026-09-25). Broken out to [`k7.7-stale-v-padding.md`](k7.7-stale-v-padding.md).


- [x] **K8 — accuracy against Kev's card** (2026-09-25). Broken out to [`k8-kev-suites.md`](k8-kev-suites.md).

- [x] **K9 — scheduling, the host head, and chunked states** (2026-09-28). Broken out to [`k9-scheduler-and-streams.md`](k9-scheduler-and-streams.md).

## Handoff

**2026-09-25, session 1 (continued).** K0–K6, **K7.1–K7.7 and K8 done**: the int8
bank is the default (1.22–1.41x on short passes, 0.8140 on transfer-v4 dev
against Kev's 0.817), and a repeated text is answered from the prefix cache
(2,269 tokens: 761 ms new, 87 ms again), bit-identically. The attention
runs on the matrix cores, the scan in the LLM's l8 shape, the MLP's gate and
up as one GEMM with SwiGLU in its epilogue, and each projection on its own
GEMM rung. Where things are:

- Oracle: `.venv/bin/python reference/dump_kev.py` (~3 min on the CPU, 17 GB),
  then `reference/dump_kev_tokens.py`. The dumps land in `reference/out/kev/`
  (~770 MB, gitignored). peft 0.21.0 was added to `.venv`.
- Tests: `go test ./kev/` runs the host tests, plus the GPU tests when not
  `-short`. The GPU tests share one staged model: ~12 s. `KEV_BANK=fp16`
  selects the control and `KEV_GEMM=<rung>` forces a GEMM rung in the
  profile. `TestGPUGEMMLadder` and `TestGPUSubmitBatching` are the K7
  measurements.
- Suite: `go build -o /tmp/kevbench ./cmd/kev && /tmp/kevbench -suite
  models/kev-suites/transfer-v4/development.jsonl -out rows.jsonl` takes
  ~40 s. Build once; do not `go run` inside a sweep.
- Server: `go run ./cmd/serve -kev -addr 127.0.0.1:8099 -token ""` is up in
  ~10 s; add `-kev-fp16` for the control.
- `KEV_ROWS=4096` gives the shared test instance room for the long-text
  latency and profile cases, which skip at the default 1024.
- `KEV_ATTN=naive` runs the control attention; `KEV_BANK`/`KEV_ATTN`
  combinations all pass `go test ./kev/`.
- `KEV_SCAN_LPC` picks the scan rung; `TestGPUScanLadder` measures them.
- `TestGPUProjLadder` prices every GEMM rung per projection; `GEMM`/`GLU`
  (fields) force a rung.
- `cmd/kev -batch N` runs a suite N records a pass; `-kev-batch` sizes the
  server's batches.
- K8: `/tmp/kevbench -suite models/kev-suites/<suite>/<partition>.jsonl
  -batch 8 -out <rows>` (~30 s to 1m45 a partition), then
  `reference/kev_suite_fp32.py` for Kev's fp32 rows (`--sample N` and
  `--also fp16,q8` for a targeted sample; a full decision-v7 partition is
  ~35 min alone, far longer beside GPU runs) and `-compare` to pair them.
  Rows are in `reference/out/kev/suites/`.
- Every path is now bit-exact across batching, chunking and cache hits,
  with either attention (K7.7). A new K-cache region needs its padding
  zeroed the same way.
- Next: nothing is known to be wrong. K9 (2026-09-28) added the scheduler
  (`-kev-batch-tokens`), the parallel head and chunked streams
  (`-kev-chunk`), measured served with `cmd/kevload` (build it and the
  server to the scratchpad and restart the server per arm). The open levers
  are the int8 bank's −0.19 pp (K8) against its 1.2–1.4x, and speed: a
  long pass is ~70% GEMMs at ~35 TFLOPS, and a short request is 44 ms of
  kernels bounded by the int8 GEMMs' read rate, plus ~2.5 ms of host.
- **Deployed 2026-09-25:** `-kev` is on the live `ai.service` line
  (small-verticals machine, beside embed/tts/stt/image). Kev stages in
  9.5 s of the ~54 s startup. Over HTTP the README ticket is 54 ms new and
  47 ms again, within 0.0018 of Kev's fp32, and `/v1/models` lists
  `kev-latest` and `jev-latest`.
