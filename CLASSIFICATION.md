# CLASSIFICATION — typed decisions with calibrated probabilities (Kev)

> **Live tracking and session handoff doc, opened 2026-09-25.** Stage letters
> are **K**. When the vertical closes, this file is frozen to
> `research/classification-vertical.md` like the others and `TODO.md` gets
> the one-line summary. Until then: tick a stage when its gate passes, put
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

  - [x] **K7.1 — the int8 bank** (2026-09-25). `kev.BankQ8`, now the
    default: int8 in the 16×16 fragment tiling with one fp16 scale per 32 k
    of a row, the LLM's L8 layout, read by the LLM's own
    `llm_gemm.comp -DQ8B` plain arm (`llm_gemm_q8_m{2,4,8}`), unchanged.
    **3.8 GB against 7.1.** Two differences from the LLM's bank, both
    deliberate. First, this is a real quantisation of bf16+LoRA weights, not
    a re-derivation of a Q8_0 checkpoint, so q is rounded against the *fp16*
    scale the kernel multiplies by. Second, every Kev pipeline now declares
    the LLM's 256-byte push range, because one command buffer takes one
    range. The dit_common kernels read its first 88 bytes. `-kev-fp16` (the
    server) and `KEV_BANK=fp16` (the tests) run the control.

    **Accuracy** (`cmd/kev`, transfer-v4 development, 764 questions):

    | | fp16 | int8 |
    |---|---|---|
    | accuracy, 656 clean | **0.8155** (535) | 0.8140 (534) |
    | Kev's published, fp32 | 0.817 (536) | |
    | argmax agreement with fp16 | – | **762/764** |
    | max / mean max \|dp\| against fp16 | – | 0.027 / 0.0024 |
    | fixtures against Kev's fp32 (max \|dp\|) | 0.0004 | 0.0077, no flips |
    | per-layer relative error against Kev | 2.0e-3 | 2.8e-2 |

    That is inside Kev's own bf16-vs-fp32 serving noise (≤0.03, about one
    flip in 300).

    **Speed**, whole passes, best of three (`TestGPUGEMMLadder`):

    | tokens | int8, best rung | fp16, best rung | |
    |---|---|---|---|
    | 37 | 35.2 (q8m2) | 49.7 (reg64) | 1.41x |
    | 80 | 42.1 (q8m2) | 54.6 (reg64) | 1.30x |
    | 101 | 46.3 (q8m4) | 56.3 (reg64) | 1.22x |
    | 494 | 155.3 (q8m4) | 154.2 (wg128x256) | 1.0x |

    Over the suite: **62.9 → 49.6 ms a record**. Through HTTP, the README
    ticket: **64 → 54 ms**. The schedule is `q8m2` up to 88 rows and `q8m4`
    above; `q8m8` never won. A 494-token pass is 3.5 TFLOP, so it is
    compute-bound at ~23 TFLOPS, and width does not matter there.

    **Why not 2x, and three kernels that did not help.** The int8 GEMMs read
    at ~125 GB/s even at 37 tokens. Three Kev-owned variants of the
    kernel were built and laddered at 37–494 tokens, and **each lost to the
    LLM's rungs at every length**:
    1. **split-K** across the grid, with a reduce (48–120 ms at 101 tokens);
    2. **several waves sharing one unpacked slab** (BM = 64–128 with 2–8
       waves: 59–98 ms);
    3. **register pipelining** (the next slab's words loaded before this
       slab's math, BK = 64): worse again.

    What they share is that each changed the per-k-step structure without
    moving its cost. The honest reading is that the kernel's time is in its
    per-k-step LDS round trip (unpack, barrier, fragment load, barrier), and
    none of these changed that. They were deleted, not kept; this table is
    their record. A kernel that reads the int8 fragments without the LDS
    round trip (an int8 WMMA path, if the device's `iu8` shape can carry the
    scale) is the next thing to try, not another rearrangement of this one.
    Also measured: submitting 8 layers a command buffer instead of 1 is 53.1
    → 51.9 ms wall, so the fence waits were never the gap (`LayersPerSubmit`,
    8 by default, 1 past 2048 tokens for the watchdog).

  - [x] **K7.2 — the prefix cache, and requests longer than a pass**
    (2026-09-25). A pass is now one of three kinds (`kev/gpu.go`, `plan`):
    the state plus as many branches as fit, the state alone, or **branches
    alone against the working state**. The working state is the attention
    KV in cells `[0, Ls)`, each GDN layer's final S, and its conv tail.
    `kev_gdn_prep` now writes that tail: the state's last three conv inputs.
    `kev/cache.go` keeps states across requests in fixed slots in their own
    buffer, keyed by the state's exact token ids and evicted LRU. Saving and
    restoring are device copies (`shaders/kev_copy.comp`), because the host
    reads these arenas at ~0.2 GB/s. The GDN layers' S and tails are one
    contiguous 52.7 MB block, so each direction is one copy plus K and V
    per attention layer (64 KB a token). Defaults: **4 states of up to 4096
    tokens** (309 MB a slot, 1.2 GB; since K7.3's fp16 KV, 181 MB and 0.7 GB), via `-kev-cache` and
    `-kev-cache-tokens`. The attention cache holds a state plus a full pass
    (`cells = rows + 8192`).

    **Gate passed.** A hit is **bit-identical** to the miss
    (`TestGPUPrefixCache`, both banks), eviction is LRU, and a request cut
    into several passes is **bit-identical** to one pass
    (`TestGPUChunkedPasses`, and over HTTP at `-kev-tokens 384`, where the
    494-token fixture takes several passes). A request longer than
    `-kev-tokens` no longer gets a 422. Only a state or a single question
    that does not fit a pass does.

    **Kev's serving table, on this device** (`TestGPUCacheLatency`, wall
    clock, best of three, int8):

    | | new text | same text again | Kev L40S | Kev H100 |
    |---|---|---|---|---|
    | 6 questions, short text (22-token state) | 76.4 ms | 74.2 ms | 41.5 / 27.7 | 18.1 / 12.9 |
    | 5 questions, 2,269-token text | 1424.8 ms | **129.8 ms** (11x) | 145.2 / 43.0 | 89.4 / 22.5 |

    Over HTTP, the 324-token fixture goes from 169 to 80 ms on a hit. A
    short state gains nothing, because its cost is the weights. **A new long
    text is where this device is furthest behind**, and the profile says why
    (`TestGPUProfile` with `KEV_ROWS=4096`, 2,439 tokens, 1375 ms of
    kernels): **the attention is 637 ms (46%)**. It is K4's correctness
    kernel, walking keys one at a time, and it is quadratic. The GEMMs are
    516 ms, ~34 TFLOPS, which is fine. The state scan is 81 ms. So K7.3
    starts with the attention.
  - [x] **K7.3 — the attention on the matrix cores** (2026-09-25).
    `shaders/kev_attn_wmma.comp` is a flash attention on 16×16×16 fp16
    cooperative matrices, laid out after `dit_attention_wmma.comp`. It uses
    one wave per (16 query rows, q head), head dim 256, 64-key blocks, and
    exp2 with log2(e) folded into q. The per-row max and rescale are staged
    through LDS as row-replicated tiles, so they are element-wise
    accumulator operations. **Kev's mask is per (row, key)**, because a
    16-row tile can straddle the state and a branch, or two branches. Blocks
    wholly inside other branches' keys are skipped. K and V moved to an
    **fp16 cache** in the fp16 arena, which halves a cached token to 32 KB
    and frees ~1 GB of the fp32 arena. q is written twice, in fp32 for the
    naive kernel and in fp16 for this one. The naive kernel stays as the
    control (`Attention = "naive"`, `KEV_ATTN=naive`) and reads the same
    fp16 cache.

    | 2,439-token pass | naive | WMMA |
    |---|---|---|
    | attention | 549.5 ms | **105.9 ms** (5.2x) |
    | all kernels | 1285 ms | 848 ms |
    | 5 questions, 2,269-token text, new / again (wall) | 1338 / 125 ms | **861 / 103 ms** |

    At 101 tokens it is 0.51 ms against the naive 0.19, a third of a
    millisecond. The kernel is not switched by length, because a cache hit
    (a short branch-only pass) would then run a different kernel from its
    miss and lose bit-identity. Accuracy is unchanged: fixtures within
    0.0079 of Kev's fp32, no flips.

    **Exactness, and the one residue.** A flash attention's bits depend on
    how keys fall into blocks. At first, removing a question moved the
    others' answers by ~1e-4, because later branches' keys shifted between
    blocks. The fix: **every branch now starts on a 64-aligned cache
    cell**. Each row's cell is the fourth word of its metadata, and the
    planner splits a pass whose padding would overflow `cells = 8192 + 2 ×
    rows`. A key block then holds the state's keys or one branch's, so a
    row sees the same blocks wherever its branch sits, and a block it may
    see none of is an exact no-op. With that, **isolation and cache hits are
    bit-exact again** with the WMMA kernel, on both banks.

    One residue is left. In one of four split layouts of the README ticket,
    one question moves by **≤ 2.5e-6**. Which bank shows it depends on
    whether the attention's epilogue writes its tile's pad rows (fp16 if
    not, int8 if so); it is exact with the naive kernel. Three causes are
    **ruled out by tests**:
    - key-block alignment (fixed above);
    - GEMM pad-row leakage (`TestGPUGEMMPadRowsDoNotLeak`: every rung of
      both banks, with pad rows of zeros, 30000 or NaN, changes 0 of 464,128
      real outputs);
    - reads of unwritten scratch (`TestGPUReadsOnlyWhatItWrote` NaN-poisons
      every per-pass tensor before each pass; every readout stays finite).

    The cause is not found. The chunked test holds the WMMA kernel to 1e-5
    and the naive one to the bit. The fp32 arena is now zeroed at
    allocation, and the epilogue writes only the pass's rows. **Found and
    fixed in K7.7:** stale V in the alignment padding, not the layout.
  - [x] **K7.4 — the GDN scan** (2026-09-25). `kev_gdn_scan.comp` now has
    the LLM's `llm_dn_scan` shape. LPC lanes share a state column, each
    holding 128/LPC of its elements in registers. Each lane loads its q and k
    elements straight from memory every token, with no LDS and no barriers,
    and the two per-column dot products are a short loop plus a
    `subgroupClusteredAdd`. A head is 2·LPC workgroups instead of one; the
    state layout and the segment semantics are unchanged.
    `TestGPUScanLadder` (state scan + branch scan, summed over 24 layers):

    | pass | K4's scan | l1 | l2 | l4 | **l8** | l16 |
    |---|---|---|---|---|---|---|
    | 101 tokens | 0.73 + 1.64 | 1.74 + 4.64 | 0.50 + 1.35 | 0.44 + 1.24 | **0.36 + 1.05** | 0.50 + 1.31 |
    | 494 tokens | 10.3 + 4.3 | 24.9 + 11.1 | 7.66 + 3.04 | 6.69 + 2.84 | **4.30 + 2.13** | 4.57 + 2.50 |
    | 2,439 tokens | 80.9 + 4.8 | 207.8 + 12.7 | 70.0 + 4.3 | 62.8 + 4.0 | **35.0 + 2.6** | 37.0 + 2.9 |

    **l8 wins at every length**, as it did for the LLM, and is the default
    (`ScanLPC`, `KEV_SCAN_LPC`). Everything else passes unchanged in all
    three configurations. transfer-v4 dev is still **0.8140**, with argmax
    agreement **764/764** against the K7.1 run (mean max |dp| 0.00012), at
    46.5 ms a record, down from 49.6. A new 2,269-token text is **821 ms**
    (was 861) and **101 ms** again. The scan is still a serial walk over the
    state's tokens (35 ms at 2,269, about 0.6 µs a token a layer). Past this
    it would take the chunked (WY) form of the delta rule, which is a
    different algorithm, not a tuning.

    **Where a 2,439-token pass goes now** (799 ms of kernels): the GEMMs are
    522 ms (in proj 152, gate 105, up 110, down 110, out proj 46) at ~34
    TFLOPS. Attention is 102. GDN prep is 44, one workgroup a token doing
    the conv, the l2 norms and the gates. SwiGLU is 40, a memory-bound
    pass over gate and up that a GEMM epilogue could absorb. The scan is 35.
    The next honest levers are that SwiGLU fusion and the prep kernel, about
    10% together, and then only the GEMMs, which are near what this device
    gives a 4B model at this length.
  - [x] **K7.6 — the SwiGLU fusion, the GEMM grid order, and the GDN prep**
    (2026-09-25).
    - **SwiGLU in the gate+up GEMM** (`shaders/kev_gemm_q8_glu.comp`, int8
      bank). The host interleaves gate and up a 32-row group at a time into
      one [18432, 2560] projection, so a 64-column tile holds both halves of
      32 outputs. The epilogue computes `silu(g)·u` element-wise on the two
      accumulators and stores fp16 straight into `down`'s A operand (a second
      slab, `hMLP`, since the GEMM reads `hA`). The swiglu pass, both fp32
      writes and a second read of A are gone. The fp16 bank keeps the
      unfused path as the control. **It is not bit-identical to the unfused
      path, as first claimed.** Two shaders compiling the same fp32
      expression need not round alike (GLSL's division and `exp` are not
      exact), and the suite moves by up to 0.0019, with argmax 764/764.
    - **The finding that mattered more: grid order.** At first the fused GEMM
      was *slower* than the pair it replaced (0.58 against 0.48 ms a layer
      at 101 rows). `llm_gemm` runs every column tile of row block 0 before
      row block 1, so a matrix read by two row blocks comes from DRAM twice
      unless it fits in the 32 MB MALL. Gate or up alone (23.6 MB) fits;
      fused (47 MB) does not, and nor does the 32 MB input projection. So
      `kev_gemm_q8_glu` runs **row-block fastest** (x = row block), and
      `-DPLAIN` builds the same kernel with an fp32 store for the other
      projections (the `rb` rungs). On the fused weight at 494 rows: `q8m2`
      3.18 → `rbm2` 1.56 ms, `q8m4` 1.81 → `rbm4` 1.26. The GLU epilogue
      costs nothing on top.
    - **A rung per projection** (`q8For(rows, n)`, `gluFor`;
      `TestGPUProjLadder` prices each). The row-block-fast grid wins on the
      wide input projection up to ~1k rows, and `llm_gemm`'s order wins on
      the 2560-wide ones (out, down) at every length and on the wide one at
      long passes:

      | kernel ms | 101 tokens | 494 | 2,446 |
      |---|---|---|---|
      | in proj | rbm4 8.7 | rbm4 27.9 | q8m8 148.4 |
      | out proj | q8m2 4.1 | q8m4 10.0 | q8m4 45.9 |
      | down | q8m2 8.6 | q8m4 20.8 | q8m4 117.0 |
      | gate+up+swiglu | glum2 16.9 | glum8 40.2 | glum8 199.9 |

    - **The GDN prep: 43.1 → 42.0 ms**, from giving the l2-norms all 256
      threads (8 a head, a clustered sum) instead of 32 serial walks. A
      rewrite with 32 contiguous channels a thread (no LDS, 4-lane norms)
      was **5x slower** (217 ms: a lane walking its own cache line is
      uncoalesced) and was reverted. Its time is the conv's history and
      weight reads. Two tokens a workgroup would need 64 KB of LDS.

    **Result** (int8, wall clock or kernel time as labelled):

    | | before K7.6 | after |
    |---|---|---|
    | kernels, 101-token pass | 46.8 ms | **42.9 ms** |
    | kernels, 494-token pass | ~147 ms | **125.4 ms** |
    | kernels, 2,439-token pass | 799 ms | **744 ms** |
    | 2,269-token text, new / again | 821 / 101 ms | **761 / 87 ms** |
    | README ticket over HTTP, new / again | 54 ms | **51.9 / 45.8 ms** |
    | transfer-v4 dev, a record | 46.5 ms | **45.0 ms**, accuracy 0.8140 |

    The ladder test now clears the prefix cache before each run, since it
    had been timing cache hits after K7.2.
  - [x] **K7.5 — batching across requests** (2026-09-25). One pass can
    hold several requests, each with its own state:
    - **State slots.** The GDN block is 8 slots (`Options.Slots`, 52.7 MB
      each), and request i of a pass uses slot i.
    - **Cells.** Each request's state starts on a 64-aligned attention cell,
      as every branch already did.
    - **Metadata, 8 words a row:** position, state or branch, segment start,
      cell, the request's state cells `[sLo, sHi)`, slot, and its state's
      end row. The mask is `key ≤ cell && (key ∈ [sLo, sHi) || key ≥ lo)`,
      with no single global state.
    - **Scan segments carry their slot.** One dispatch runs every state in
      the pass, one every branch.
    - **Cache copies take a work slot and a base cell,** so a hit restores
      into whatever slot its request got.
    - **The WMMA kernel.** It skips a key block that no row of its tile may
      see, and leaves a row that has seen no key yet un-rescaled.
    - **`ForwardBatch`/`ProbsBatch`** plan one pass when rows, cells and
      slots allow, and otherwise run the requests one at a time.

    The backend runs one worker that takes whatever is queued when the device
    frees up, up to `-kev-batch` (8), and never waits for more.

    | | one at a time | batched |
    |---|---|---|
    | 8 short requests, `TestGPUBatchThroughput` | 401 ms | **266 ms** (1.51x, 30 req/s) |
    | transfer-v4 dev, `cmd/kev -batch 8` | 45.0 ms a record | **27.3 ms** (1.65x) |
    | 16 concurrent HTTP requests | 822 ms, 19.5/s | **553 ms, 28.9/s** |
    | 1 HTTP request | 52 ms | 55 ms |

    **Accuracy.** The suite batched is **0.8140**, with argmax **764/764**
    against unbatched. The naive attention is **bit-exact** in any batch,
    cold, cached or half-cached. With `kev_attn_wmma`, a request's
    probabilities move with its partners in the pass by up to **6e-4** (mean
    3e-5) over the suite, and up to 1.7e-4 on the fixtures. That is the
    same unexplained position effect as K7.3's residue, larger here
    (**K7.7 found it: stale V, not position; batches are now bit-exact**). Forcing
    one GEMM rung does not change it, and the naive kernel removes it. Kev's
    own batched server makes the same promise, "up to floating-point
    reassociation". `TestGPUBatchMatchesSingle` holds WMMA to 1e-3 and naive
    to the bit. Bug found on the way: the metadata and segment tables
    outgrew their arena regions (8 and 4 words), and the first batch hung
    the GPU. They are now sized for 8 words a row and 4 a segment.


  - [x] **K7.7 — the WMMA residue was stale V** (2026-09-25). K7.3's
    ≤2.5e-6 across chunked layouts and K7.5's ≤6e-4 across batch partners
    had one cause, and it was not the layout. **A request's answers depended
    on what had run before it.** Bisected with `run` over the first L layers
    and the batch planner:
    1. Alone and batched agree when both run on a fresh device. An alone run
       after *any* batched pass is off in one row (the README ticket's
       question-2 `<decide>`), from the first attention layer on.
    2. On that row the WMMA output moves by one fp16 ulp in one element.
       Clearing **V in cells 143–191** restores it; K there, and K or V in
       a block the row sees none of (even filled with ±8192), change
       nothing.
    3. Cells 143–191 are the alignment padding after the row's own branch
       (cells 128–142), inside the 64-key block it partly sees. The row
       masks them, so P is exactly 0 there, but 0 × a stale V still moves
       the matrix cores' sum. Every segment ends in such padding, and it
       holds whatever an earlier pass wrote. The naive kernel never reads a
       masked key, which is why it was always exact.

    The fix is in `kev_attn_prep.comp`: the last row of each segment
    zeroes V from its cell to the next 64-cell boundary. The first branch
    row of a request whose state is resident (a cache hit, or a later pass
    of a long request) zeroes the state's padding. Zero is what a fresh
    device holds. It costs nothing measurable (45.0 and 27.3 ms a record,
    unbatched and batched, as before).

    **Gate passed.** transfer-v4 dev in batches of 1 and of 8 is
    **byte-identical** (was up to 6e-4 apart), and accuracy is unchanged at
    0.8140. `TestGPUBatchMatchesSingle` and `TestGPUChunkedPasses` now hold
    the WMMA kernel to the bit, as they did the naive one.
    `TestGPUStaleCacheDoesNotLeak` fills every attention layer's K and V
    with junk before cold, batched and cache-hit runs and requires the same
    bits as over a zeroed cache. Without the fix it fails by up to 2.6e-4;
    over a *zeroed* cache, batched matched alone even before the fix. All
    four `KEV_BANK`/`KEV_ATTN` combinations pass. On the way,
    `TestGPUGEMMPadRowsDoNotLeak` gained the K7.6 `rb` rungs (on their own
    row-block-fastest grid) and a pad of ordinary activations. It had
    predated them, and they do not leak either.

 (2026-09-25).
  `cmd/kev -suite` runs one of Kev's frozen suites and scores it as
  `kev.benchmark` and `kev.metrics` do: argmax, Brier and ECE over the clean,
  knowable rows, raw Brier from the served p (p^T renormalised is
  softmax(z)), and the unknowable share at p ≥ 0.9. `-compare` pairs two row
  files. `reference/kev_suite_fp32.py` writes the same rows from Kev's own
  code in fp32 on the CPU (`model.probs`, ~1.3 s a transfer-v4 record, ~5 s a
  decision-v7 one; it resumes an interrupted file). The suites are copied
  from Kev's repo at `9fdf054` into `models/kev-suites/{transfer-v4,
  decision-v7, transfer-v9}` (development and test). Rows land in
  `reference/out/kev/suites/` (gitignored).

  **The oracle reproduces Kev's card to the digit**, so the card's numbers
  are fp32 numbers and the comparison below is like for like:

  | | Kev's card | Kev fp32, here | fp16 bank | int8 bank (served) |
  |---|---|---|---|---|
  | transfer-v4 dev, accuracy (656) | 0.817 | **0.8171** | 0.8155 | 0.8140 |
  | transfer-v4 dev, Brier served / raw | 0.243 / 0.269 | **0.2430 / 0.2691** | 0.2430 / 0.2691 | 0.2422 / 0.2681 |
  | transfer-v4 dev, ECE | 0.042 | **0.0416** | 0.0425 | 0.0396 |
  | transfer-v4 test, accuracy / Brier (656) | 0.838 / 0.224 | **0.8384 / 0.2242** | 0.8384 / 0.2241 | 0.8384 / 0.2238 |
  | decision-v7 dev (1,264) | 0.873 | sampled | **0.8726** | 0.8687 |
  | decision-v7 test (1,200) | 0.865 | – | **0.8650** | 0.8642 |
  | transfer-v9 dev, MMLU-Pro (200) | 0.565 | sampled | **0.565** | 0.560 |
  | transfer-v9 dev, unknowable at ≥ 0.9 (110) | 0.00 | sampled | 0.00 | 0.00 |

  decision-v7 dev's raw Brier on fp16, **0.2037**, is also the raw Brier
  `head.json` records for the temperature fit on those same 1,264 rows. The
  card's transfer-v9 "0.490" and "buried 0.67" are the *previous*
  checkpoint's; the current one publishes only MMLU-Pro and the unknowable
  share. (Here buried is 0.700 on both banks; transfer-v9 dev overall 0.7734
  fp16, 0.7715 int8.)

  **Row by row against Kev's fp32** (`-compare`):

  | | fp16 bank | int8 bank |
  |---|---|---|
  | transfer-v4 dev, argmax agreement | **763/764** | 761/764 |
  | transfer-v4 dev, max / mean max \|dp\| | 0.0030 / 0.00015 | 0.027 / 0.0024 |
  | transfer-v4 test, argmax agreement | **764/764** | 764/764 |
  | transfer-v4 test, max / mean max \|dp\| | 0.0011 / 0.00012 | 0.030 / 0.0023 |
  | decision-v7 dev, sampled: argmax agreement | **399/400** | 395/400 |
  | transfer-v9 dev, sampled: argmax agreement | **155/155** | 148/155 |

  The last two rows are a sample, not a full run (decision-v7 is ~1.7 s a
  record in fp32): a seeded random 150 records (`--sample 150`), every
  record where the fp16 and int8 banks disagree (`--also`, 6 and 7 of them),
  and on decision-v7 the first 183 records of an interrupted full run. The
  flips are chosen on purpose, so int8's agreement there is not a rate.

  **fp16's disagreements are ties in fp32**: the top two options are within
  0.000–0.002 of each other, on all four partitions. The fp16 bank matches
  fp32 to about Kev's own isolation-gate tolerance (1e-3), and it lands on
  every published accuracy exactly or within one question. **int8's are
  not all ties**: fp32 margins ≤ 0.002 on transfer-v4, but up to 0.016 on
  decision-v7 (sst5, yelp) and 0.032 on transfer-v9's 10-way MMLU-Pro,
  where three of its four flips land on option 8 (too few to call a
  pattern).

  **What int8 costs.** It is a real quantisation, so its flips are not
  symmetric noise. Against fp16 over the five partitions with an fp16
  control it is **−9 questions of 4,822 (−0.19 pp)**: transfer-v4 dev −1
  (−2 against fp32), test 0, decision-v7 dev −5 (1462/1468 argmax agreement, all six flips at
  p ≈ 0.49 against 0.49), decision-v7 test −1, transfer-v9 dev −2. Calibration
  does not move (Brier within 0.001). That is smaller than the gap between
  Kev's bf16 server and its fp32 (~0.03 in p). It buys 1.2–1.4x on short
  passes (K7.1); `-kev-fp16` is the exact option and costs 3.3 GB more.

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
- Next: nothing is known to be wrong. The open levers are the int8 bank's
  −0.19 pp (K8) against its 1.2–1.4x, and speed: a long pass is ~70% GEMMs
  at ~35 TFLOPS, and the README ticket is 52 ms, bounded by the int8 GEMMs'
  read rate. `-kev` is not in ai.service yet.
