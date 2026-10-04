# RUNE — the classifier, replacing Kev

> **Live tracking and session handoff doc, opened 2026-10-03.** Stage letters
> are **R**. It replaces [`classification-vertical.md`](classification-vertical.md)
> (Kev-4B, K-stages), which is frozen as Kev's record. Tick a stage when its
> gate passes, put its measured numbers under it, and keep **§ Handoff** at the
> bottom current.

## What we are building

`GOALS.md` item 6, classification, on a stronger model. Kev-4B (Qwen3.5-4B +
LoRA + pointer head, TypeSafe System One) is a long way behind on the current
benchmarks. **Rune v3** (Invergent AI, Apache 2.0) is second on Decision Index
0.2.1 at 57.44, 0.45 behind Jev 1.13; Kev was Jev's open approximation.

- Weights: `surogate/rune-26b-a4b-GGUF` on HF. **Despite the name the repo
  holds bf16 safetensors only** (11 shards, 51.6 GB), plus the tokenizer, the
  chat template and the vision processor. The repo is gated.
- Protocol: surogate's **decisions v1**,
  `invergent-ai/surogate docs/inference/decisions.md`: `POST /v1/decisions`
  (aliases `/api/alpha/decisions`, `/api/v1/decisions`), OpenRouter's field
  names. It is stable and pinned by golden fixtures, so we have an exact
  target.

## The model: Gemma 4 26B-A4B (`gemma4_text`)

Nothing in this repo runs Gemma yet (the `gemma` hits in `llm/` are
`GEMM`-at-decode). The text tower, from transformers 5.17.0
`models/gemma4/modeling_gemma4.py`:

| | |
|---|---|
| layers | 30; `layer_types` sliding ×5 then full, repeating (5 full layers) |
| hidden | 2816, `rms_norm_eps` 1e-6, embedding × √2816, tied head |
| vocab | 262,144 |
| sliding attention | 16 q heads, 8 kv heads, head 256, window 1024 (key position > query position − 1024, on **positions**, not Kev's packed cache cells), NeoX RoPE θ 1e4, inv_freq_j = θ^(−2j/256) |
| full attention | 16 q heads, 2 kv heads, head **512**, **K = V** (no v_proj: V = unscaled-RMS(k_proj(x)), K = RoPE(q-style-norm(k_proj(x)))), "proportional" RoPE θ 1e6: NeoX rotate-half over all 512 (pairs j, j+256), inv_freq_j = θ^(−2j/512) for j < 64 and **0 for j ≥ 64** (cos 1, sin 0), so one table-driven RoPE serves both kinds |
| attention | q_norm, k_norm (scaled RMS), v_norm (unscaled RMS), **scaling 1.0** |
| dense MLP | GELU-tanh, 2112, every layer |
| MoE | 128 experts, top 8, expert inner 704, GELU-tanh; router on the *residual* (unscaled RMS × scale × 2816^-½, softmax, top-k, renormalise, × per_expert_scale) |
| layer | in-norm → attn → post-attn norm → +res; pre-ff norm → MLP → post-ff-1; pre-ff-2 → MoE → post-ff-2; sum → post-ff norm → +res; × layer_scalar |
| head | final norm, logits/30 → tanh → ×30 (softcap **is** applied before the letter readout: surogate `PATCHES.md:3461`) |

RMSNorm is plain `x * w`, not Gemma 2/3's `x * (1 + w)`. A second reference
exists: surogate's CUDA `csrc/src/serve/targets/gemma4_moe/`.

**Footprint.** About 25.8B of the 26.5B parameters are the text tower; the vision
tower is ~0.5B. At bf16 that is 51.6 GB, which is too much beside the other
small verticals. **The target is Q8 (decided 2026-10-03)**: int8 rows with a
scale each, as Kev's K7.1 and H3's M11a did, quantised from the bf16 weights
by us, so about 28 GB. Price it against bf16, not fp16, as M11a did. Two
shortcuts follow from the readout:

- **The head needs only the option letters' rows.** The answer is a softmax
  over ≤255 label logits, so the 262k × 2816 head is a gather of ≤255 rows,
  not a GEMV over 738M weights.
- **The embedding can stay on the host** (Kev did the same with its 1.27 GB
  table): 262144 × 2816 is 1.48 GB at bf16.

## The protocol (decisions v1), in brief

Each question is a two-turn chat, rendered by the model's own template with
thinking off: a fixed system prompt, then a user turn

```
SHARED STATE (JSON string):
<json.dumps(state)>

QUESTION:
<instructions>
OPTIONS:
A: <first>
...
Answer with one option letter only.
```

The answer is the softmax, in double, of the option labels' logits at the first
generated position, divided by the server's `T` (Rune's card says **T = 2**).
Choice: argmax (the first on a tie), confidence `(max − 1/n)/(1 − 1/n)`. Noul: p(true), with
A = false and B = true always. Score: `Σ i·p_i` and TypeSafe's
`score_confidence`. Past 26 options the labels come from a codebook (AA, AB…)
and the prompts say "code". The shared state is prefilled once, and each
question continues from it. **The oracle for all of this is surogate's
`fixtures/serve/decisions_v1/golden.json`** (14 cases, from an independent
stdlib-only Python implementation, `make_golden.py`): prompt text byte for
byte and answers bit for bit for fixed logits.

Surogate's **47-token suffix floor** is about *their* MoE kernels (a narrow
prefill takes different GEMMs). It is not part of the meaning of an answer. We
keep the shared-prefix rule and make our own paths bit-exact across widths,
as Kev's K7.7 did.

## Stages

- [ ] **R0 — access and weights.** The user accepts the gate on HF and logs in
  (`! huggingface-cli login`). Download to `models/rune-26b-a4b/` (51.6 GB;
  724 GB free). Diff the tokenizer and template against public
  `google/gemma-4-26B-A4B-it`.
  2026-10-03: access granted, and the download is under way (~16 MB/s,
  ~55 min). **Rune's `config.json` is identical to
  `google/gemma-4-26B-A4B-it`'s and its `chat_template.jinja` is
  byte-identical**, so the architecture table above is Rune's.
- [x] **R1 — the protocol, host only** (`decide/`). Request parsing with key
  order kept and Python's last-wins duplicates, `json.dumps` rendering
  (reuse `kev/value.go`'s `PyFloatRepr` and `llm/pyjson.go`'s rules),
  labels and codebook, the system prompts, the readout in double with `T`,
  and the response envelope. Gate: all 14 golden cases, text byte for byte
  and numbers bit for bit.
  **Done 2026-10-03.** `decide/` (`value.go`, `decide.go`), `go test
  ./decide/` passes the golden file (vendored at surogate `3f31ccac`) plus
  the refusals and the tempered readout. One thing the golden file caught:
  the readout's `exp` must be glibc's (`decide/exp.go`, cgo). Go's
  `math.Exp` is off by an ulp often enough to move 5 of the 31 golden
  answers, and both other implementations call glibc.
- [x] **R2 — the tokenizer and the chat template.** Gemma 4's tokenizer.json
  (SentencePiece-style BPE with byte fallback; check whether
  `zimage/tokenizer` covers it) and the thinking-off render
  (`<|turn>system … <|turn>user … <|turn>model\n<|channel>thought\n<channel|>`
  — confirm against `chat_template.jinja`). Gate: token ids identical to HF
  `apply_chat_template` + tokenizer over the golden prompts plus a
  Romanian/emoji corpus, and every label a single token *in context*.
  **Passed 2026-10-03.** Rune's `tokenizer.json` and `tokenizer_config.json`
  are byte-identical to google/gemma-4-26B-A4B-it's
  (`models/gemma-4-tokenizer/`), and `GEMMA4_TOKENIZER=../models/rune-26b-a4b`
  passes the same test.
  `gemma4/tokenizer.go` (a heap BPE: one segment is the whole user turn,
  because the pre-tokenizer's split on " " runs after the normaliser
  removed every space) and `gemma4/prompt.go` (the template, transcribed:
  `<bos><|turn>system\n…<turn|>\n<|turn>user\n…<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`).
  `reference/dump_gemma4_tokens.py` writes `gemma4/testdata/prompts.json`:
  31 golden questions plus 6 awkward states (Romanian/CJK/emoji,
  whitespace runs, byte fallback, a literal `<turn|>`, 24k chars of
  wikitext). All 37 have identical ids, every label is one token in context,
  and the codebook matches HF's at 679 codes. A 24k-char state encodes in 11
  ms. The golden file's 255-option case uses a synthetic codebook, so the
  dump re-labels extended questions with the real one.
- [x] **R3 — the oracle.** `reference/dump_rune.py`: transformers 5.17.0
  Gemma4 text tower in bf16 on the CPU (~52 GB of RAM: nothing else big may
  run), per-layer hidden states and the label logits for the golden prompts
  and a few long states (> 1024 tokens, to cross the sliding window). Dumps
  to `reference/out/rune/` (gitignored).
- [x] **R4 — loading and the Q8 bank** (`gemma4/`). Safetensors → int8
  rows with a scale each: attention projections and the dense MLP on
  llm_gemm's Q8 rungs (as K7.1); **experts as Q8 for the grouped MoE GEMM**
  (`llm/` ships its experts at Q4 from GGUF, so the int8 grouped path is new
  or adapted); the router and norms fp32; a bank cache like M11b's so
  staging does not re-quantise. Gate: the bank dequantises to within int8
  rounding of bf16, and its size is measured (~28 GB expected).
- [x] **R5 — the GPU forward.** Gemma 4 layer kernels: the norms, q/k norm
  plus RoPE (default and proportional-partial), sliding-window attention
  (window 1024) and full attention at head 512 with K = V, GELU-tanh MLP,
  router, experts, combine, softcap, and the label-row head. Start from
  Kev's WMMA flash attention (K7.3) and the LLM's MoE path. Gate: per-layer
  against R3. Then the label probabilities: bf16-on-GPU against the oracle
  to float noise, and Q8 within a stated bound.
  **First block gate passed 2026-10-03: the MoE.** `llm.MoEGPU` at Rune's
  shapes with `WithMoEGELU()` (new: `-DACT_GELU=1` builds of the seven Q8_0
  up rungs, `shaders/llm_moe_gemm.comp`'s `GATE_ACT` macro; the LLM's own
  builds are byte-identical, checked by recompiling three of them) and the
  zero-down 64-wide shared expert. `TestMoEBlockGemmaShapes` (random
  weights, 100 tokens, against a float64 CPU reference over the same fp16
  input, router and dequantised banks): 800/800 routing choices agree, and
  the output is within 4.5e-4 relative RMS. The block must be pinned to the
  GEMM (`PinGemv(true)`): the GEMV rungs are compiled for K ≤ 2560 and have
  no GELU build, and a GELU block refuses them in `graph`.
  **Second block gate passed 2026-10-03: attention.** `shaders/gemma4_attn_prep.comp`
  (norms, table RoPE, K/V cache writes, V padding zeroed) and
  `shaders/gemma4_attn.comp` (Kev's flash attention with the window on
  positions, no gate, scale 1; head 512 streams Q and splits V's columns
  over y = 2 x head, since a sequence shares one z extent). `TestAttentionKinds`:
  a 1,100-token state (past the 1,024 window) plus two branches at
  64-aligned cells, against float64 HF semantics. Worst |o − ref|:
  1.7e-3 (sliding), 2.2e-3 (full). Over 1,170 rows one layer takes 2.5 ms
  sliding and 11.0 ms full, so the full kernel is the first speed item (R8).
  **R5 passed 2026-10-04: the whole model against HF fp32** (`TestOracle`,
  6 prompts, 122–5,500 tokens). Every prompt chooses the reference's option,
  and p(T=1) is within 0.042. **Bf16 itself moves the same prompts by up to
  0.021** (`dump_rune.py --bf16`, Rune as served upstream: urgency 0.5068 →
  bf16 0.5273 → Q8 0.5464; product A 0.1456 → 0.1270 → 0.1873; tone 0.9879
  → 0.9873 → 0.9844), so the gate is "same choice, within 0.05". Teacher
  forced (`TestOracleTeacher`), each layer is 0.3–1.4% relative RMS except
  the last (layer 29, full) at 3.5%: the first suspect if R7 wants
  accuracy back (keep it fp16, as M11a kept layers). Free-running, the
  residual is 9–15% at layer 29 for a 0.07–0.4 error in the label
  logits.
  **The bug the first run found: don't fold the pre-FFN norms into int8.**
  `pre_feedforward_layernorm(_2)` weights span 6e-6 to 266 within a layer.
  Folded into the matrices' columns, they put tiny and huge columns in one
  32-weight group, and the FFN half came out 20–25% wrong a layer
  (`TestOracleInternals` located it: attention 0.7%, dense MLP 20%,
  experts 25%, routing right). They are now applied in `gemma4_post.comp`.
  The router (fp16, a per-element exponent) carries scale · 2816^-½ / w_pre2,
  with a staging check that it fits fp16. The design section's fold
  paragraph is superseded by this.
  A 5,500-token prompt is 6.5 s; a 125-token one 0.20 s (one submit a
  layer for the MoE, the full attention at 11 ms, the router GEMM). Staging
  plus the six prompts is ~32 s.
- [x] **R6 — serve it.** `cmd/serve -rune` answers `/v1/decisions` and the
  two aliases: the shared prefix prefilled once, the question suffixes as
  batched rows over it (Kev's K7.2 prefix cache and K7.5/K9 scheduler
  carried over), `-rune-temperature` default 2, and `/v1/models` lists it.
  Gate: the README-style ticket over HTTP, and the golden cases end to end
  with our logits.
  **Written 2026-10-03, not yet run on the model:** `api/decisions.go` (the
  three paths, v1's error envelope with `param` and `code`, 404 for an
  unknown model), `backend/rune.go` (one staged model behind
  `/v1/decisions` and `/v1/systemone`; model ids `rune`, `rune-26b-a4b`,
  plus `kev-latest` and `jev-latest` for Kev's clients), `cmd/serve -rune`
  (`-rune-model`, `-rune-rows`, `-rune-temperature` default 2; with `-kev`
  as well, Kev keeps `/v1/systemone`), and `gemma4/answer.go` (plan,
  pass, readout; `order_averaging` is implemented, `decide.AverageOrders`;
  `thinking` and images answer v1's 400 codes). `TestPlan` holds the
  shared-prefix rule on real prompts.
  **R6 gate passed 2026-10-04, served** (`cmd/serve -rune -addr
  127.0.0.1:8099`): stages in 25.6 s. surogate's doc example ticket (refund
  noul, tone choice, urgency score) answers refund 1.000, tone annoyed
  (0.786), urgency 0.93 (somewhat) at T = 2, **~170–200 ms over HTTP**
  (212 input tokens). Both aliases answer. `order_averaging` mirrors the
  choice and noul questions (input_tokens 295) and leaves the score alone.
  `/v1/systemone` answers Kev's envelope from the same model (164 ms). The
  refusals: unknown model 404 `model_not_found`, a non-boolean `thinking`
  400 with its param, System One's missing state 422. `/v1/models` lists
  rune, rune-26b-a4b, kev-latest and jev-latest.
- [x] **R7 — accuracy.** Q8 against bf16 over a held-out set (Kev's suites in
  `models/kev-suites/` map onto decisions questions), with flips counted
  by margin as in K8. Confirm T = 2 by NLL on a calibration split. Decide
  whether any layer must stay bf16 (as M11a / Q13 kept some).
  **Run 2026-10-04** (`cmd/rune -suite`, Kev's scoring: clean, knowable
  rows; rows in `reference/out/rune/suites/`, Kev's in
  `reference/out/kev/suites/`):

  | suite | Kev q8 | Rune Q8 | right in Kev only / Rune only | fitted T |
  |---|---|---|---|---|
  | transfer-v4 dev | 0.8140 | **0.8415** | 36 / 54 | 2.20 |
  | transfer-v4 test | 0.8384 | **0.8704** | 28 / 49 | 2.18 |
  | transfer-v9 dev | 0.7715 | **0.8002** | 74 / 104 | 1.93 |
  | transfer-v9 test | 0.7734 | **0.8308** | 56 / 116 | 1.84 |
  | decision-v7 dev | **0.8687** | 0.8378 | 91 / 52 | 2.12 |
  | decision-v7 test | **0.8642** | 0.8275 | 99 / 55 | 2.00 |

  Rune wins every transfer partition by 3–6 points and loses
  decision-v7, Kev's in-distribution suite, by ~3.5. The loss is spread
  over the labelled-dataset tasks Kev trained on (decision-v7 dev:
  AG News 0.775 vs 0.863, Banking77 0.838 vs 0.900, TREC 0.875 vs 0.950,
  MNLI 0.875 vs 0.925), and no task collapses. **The card's T = 2 is
  confirmed**: the NLL-fitted T is 1.84–2.20 on all six, and T = 2 takes
  ECE from ~0.10 to ~0.03 (transfer-v4 dev: Brier 0.2440 → 0.2211, NLL
  0.530 → 0.397). A suite runs at ~169 ms a record (764 in 2m09).
  **Q8 is not the decision-v7 gap.** Every 4th question of decision-v7 dev
  (367) was re-run through HF in bf16, as Rune is served upstream
  (`reference/rune_suite_ref.py`, the exact prompt ids, ~1.7 s a question
  on the CPU). Q8 agrees on 364/367 answers (the 3 flips have margins of
  0.07–0.34 in bf16), with accuracy 0.8133 against bf16's 0.8165 (one
  question of 316) and mean max |dp| 0.010. Kev beats bf16 Rune on the same
  sample too (0.8449 vs 0.8165). The gap is the model: decision-v7 is
  Kev's home suite. No layer needs to stay wider than Q8.
- [x] **R8 — speed.** Prefill rate, suffix waves, the README ticket against
  Kev's 54 ms; memory resident at idle and peak in the server
  ([[server-memory-is-not-standalone-memory]]).
  **Done 2026-10-04.** Where a short request's time goes (`cmd/rune
  -profile`, a 3-question 212-token ticket, 164 ms of GPU before R8): the
  experts are 69% (`moe.up` 75 ms, `moe.down` 38 ms) and **at the bandwidth
  roofline**. 212 tokens route 1,696 rows over 128 experts, so every expert
  is read: 16.2 GB up and 8.1 GB down a request, at ~215 GB/s. **At Q8 a
  single short request cannot go below ~115 ms** (Kev's 54 ms read a 3.8
  GB dense model). The dense int8 GEMMs are 31 ms, and their rung ladder
  (`-ladder`) puts Kev's schedule within 1–5% of the best except qkv at
  ~1024 rows (45.5 → 39.8 ms); the measured schedule is in `gemmFor`.
  What R8 changed:
  - **The stand-in shared expert's dispatches are gone** (5.4 ms;
    `llm.WithMoENoShared`). Its gated rows are zeroed by `syncShared` at
    every run. A once-only zero at construction was a stale-read bug:
    a shorter earlier pass leaves routed rows where a longer pass's
    shared rows are. `TestBatch` caught it (0.114 off). It is now
    bit-exact, batched against alone.
  - **One command buffer for 8 layers**, the MoE included
    (`llm.MoEGPU.Dispatches`; a layer at a time past 2048 rows). Bit-identical,
    and worth little: the submits were not the cost.
  - **Full attention through LDS once per GQA group**
    (`gemma4_attn_gqa.comp`: one workgroup is the 8 query heads sharing a
    KV head, K/V blocks of 16 keys in LDS). The old kernel re-read K/V per
    head and per half, ~60 GB a layer at 5,500 tokens. Full layers 1,968 →
    861 ms at 5,500 tokens; the request 4.23 → 3.08 s. `TestAttentionKinds`
    holds it to the same reference error. `GPU.Attention = "wave"` is the
    control.
  - **Batching across requests** (`gemma4.NewBatch`, `RunPlans`;
    `backend/rune.go`'s worker packs whatever waits up to
    `-rune-batch-tokens` 1024, where a row is cheapest). Bit-exact against
    alone. Served (`cmd/kevload`, a server per arm, the README ticket):

    | load | alone p50 / p90 | batched p50 / p90 | req/s alone → batched |
    |---|---|---|---|
    | burst of 16 | 1477 / 2451 ms | 802 / 1380 ms | 6.1 → 11.6 |
    | 4 req/s | 278 / 792 ms | 192 / 354 ms | 4.0 → 4.0 |
    | 8 req/s | 3285 / 4383 ms | 322 / 517 ms | 6.2 → 7.5 |

  The 212-token ticket is 158 ms alone (164 before). The forward against
  pass length: 32 rows 73 ms, 128 109, 256 148, 512 208, 1024 369, 2048 822,
  4096 2527, 8192 7979. Past 2048 attention dominates: at 5,500 tokens
  it is 54% (sliding 792 ms, full 861 ms), and both kernels run far below
  the matrix cores' rate (serial per-block mask and max work on 16 lanes).
  **Open:** an attention rework for long states, a prefix cache for
  repeated states, chunked states past 8192 tokens.
- [x] **R9 — deploy and retire Kev.** `-rune` replaces `-kev` on the
  small-verticals `ai.service` line. Remove `kev/`, `backend/kev.go`,
  `cmd/kev*` and Kev's oracle. `/v1/systemone`: see open question R-o1.
  **Done 2026-10-04.** `ai.service`'s small-verticals line runs `-rune`
  (always resident, the user's call: latency first for classification).
  Staged in 25.2 s; the service was listening ~65 s after Rune finished
  staging. Live: the ticket answers on `/v1/decisions` in ~170–215 ms, and
  the README's TypeSafe example on `/v1/systemone` (model `jev-latest`) in
  ~170 ms. Kev is removed: `kev/`, `cmd/kev`, `backend/kev.go` (+ its
  scheduler test), `-kev*` flags, its GDN/attention/copy shaders and SwiGLU
  builds, `reference/dump_kev*.py`, `convert_kev_head.py`,
  `kev_suite_fp32.py`, `reference/kev/`. Kept: `kev_gemm_q8_glu.comp`
  (Rune's rb and GELU builds), `cmd/kevload` + `reference/kev_fixtures.json`
  (the System One load generator), `models/kev-suites/` (R7's suites),
  Kev's suite rows in `reference/out/kev/suites/` (the comparison).
- [ ] **R10 — optional extensions.** `thinking` (gate 0.7, 512-token greedy
  thought, read after `<channel|>`), `order_averaging`, and images (Gemma 4's
  vision tower, `--vision`).

## The GPU design (R4/R5), worked out 2026-10-03

A layer is three kinds of work, and each already has a host in this repo:

| part | carried from | what changes |
|---|---|---|
| projections (q, k, v, o; dense gate+up, down) | Kev's int8 GEMMs (`llm_gemm.comp -DQ8B` rungs, `kev_gemm_q8_glu`) | the GLU epilogue gains GELU-tanh (`-DACT`) |
| attention | Kev's packed pass (state rows + branch rows, segment metadata) and `kev_attn_wmma` at head 256 | a sliding-window bound (1024) in the mask; a head-512 build with K = V for the 5 full layers; Gemma's q/k/v norms and two RoPEs in the prep kernel |
| routed experts | `llm.MoEGPU` (router GEMM, route, perm, grouped up/down at Q8_0, combine) | GELU-tanh in MODE 0's epilogue, and nothing else: Gemma has no shared expert, so its slot gets a 64-wide expert whose **down is all zeros**. Its contribution is exactly 0 × σ(g) = +0, added last, so the combine's sum is the 8 routed terms bit for bit, and the route, perm and combine kernels stay as they are |

**Kev's pass layout is decisions v1's shape**: one state, then each
question's branch attending to the state and to itself. So a request is one
packed pass, with no KV copies, as Kev's K5 was. The shared prefix is the
longest common token prefix (R2: merges can cross the state/question join).

**Folds, so that one unweighted RMSNorm of the residual feeds the whole FFN
half.** The router reads `rms(residual) × scale × 2816^-½`; the experts
read `rms(residual) × w_pre2`; the dense MLP reads `rms(residual) × w_pre`.
A per-column weight on a matmul's input folds into that matrix's columns,
so the loader multiplies `scale × 2816^-½` into the router's columns,
`w_pre2` into every expert's gate/up columns, and `w_pre` into the dense
gate/up columns, all before quantising. `per_expert_scale` folds into each
expert's down rows (`gemma4/experts.go`). What stays: the router's softmax,
top-8 and renormalise, which is exactly the LLM's route (Qwen's
`norm_topk_prob`); and the two output norms (post_ff_1 on the dense MLP,
post_ff_2 on the expert sum), which are nonlinear and so cannot fold. They
are why the dense MLP cannot ride in the LLM's shared-expert slot: the
combine would sum it into the experts before their norm.

**Sizes at Q8 (8.5 bits).** Experts 30 × 3 × 128 × 704 × 2816 = 22.8B weights
→ 24.2 GB, one 0.81 GB buffer a layer (the LLM's one-bank-a-layer
arrangement, `moeMaxBanks` 48 ≥ 30). Attention + dense MLP ≈ 1.9B → 2.0 GB.
Router fp16 30 × 128 × 2816 → 22 MB. The embedding stays on the host (bf16,
1.48 GB); the head is a gather of the ≤255 label rows. **≈ 26.3 GB on the
device**, plus activations.

## Open questions

- **R-o1 — settled 2026-10-03: `/v1/systemone` stays, as a translation
  onto decisions** (the user: "keep the API as a translation layer if we
  can"). It can: Kev's System One asks the same three types (noul, choice,
  score); the "pointer head" was Kev's internal readout, not an API
  feature. `decide/systemone.go`: the loose request rules (any state,
  optional instructions, a noul's sides optional) map onto the nearest v1
  question; the prompt is v1's; answers come back in Kev's envelope,
  rounded to four places, 422 on refusal. One deliberate change: score
  confidence is TypeSafe's published `score_confidence` (v1's), not Kev's
  approximation of it. `TestSystemOne` holds it (numbers checked against an
  independent Python computation).
- **R-o2 — settled 2026-10-04.** Rune stays resident (the user: latency
  first). It fits: the service is 36 GB of device memory at rest, ~98 of
  117 GB with the swap slot at its 60 GB worst (§ Handoff).

## Handoff

**2026-10-04, session 1 (end).** R0–R9 done: Rune is deployed in place of
Kev and Kev's code is gone (git history). Results are in the stages above:
R7 (accuracy against Kev and bf16), R8 (speed; the expert-bandwidth floor),
R9 (deploy).

Open, in the order they would matter:
- **Long states.** Attention is 54% of a 5,500-token pass (3.1 s). Both
  kernels run far below the matrix cores' rate: the per-16-key serial
  mask and max work, single-wave sliding workgroups. That is a kernel
  project in the G-stage style. Past 8,192 tokens a request is refused
  (no chunked states yet).
- **A prefix cache** for repeated states (Kev's K7.2). The experts still
  dominate a short pass, so this pays only on long repeated states.
- **A bank cache** (M11b-style) so a restart does not re-quantise (~20 s of
  the 25 s staging).
- **R10:** `thinking`, images.
- **Memory (R-o2), measured 2026-10-04:** the whole small-verticals
  service at rest (swap slot idle) is 28 GB GTT + 8 GB VRAM, with 38 GB of
  system RAM used and 78 GB available. The swap slot's 60 GB worst case puts
  it at ~98 of 117 GB, so Rune stays resident beside it. Re-measure if a
  vertical grows.
