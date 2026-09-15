<!-- LLM.md L2b. The hyper-connection block, CPU reference, checked against
     llama.cpp's own activations. Cited from llm/ and reference/eval_dump.c. -->

[← LLM.md](../LLM.md) · [research index](README.md) · [L2a](l2a-prefill-attribution.md) · [L0d](l0d-quant-error.md) · phase 1

# L2b — the hyper-connection block, and what the oracle is actually computing

**Result: the block L2a put at the front of the queue runs in Go and
reproduces llama.cpp value for value — four of its seven mixers to 6.2e-08
rms on a gate in [0, 1], which is f32 round-off.** `hc_init` and the
`token_embd` gather under it are **bit-exact**; the combine is 1e-09; the
F32-weighted norm and inject are 1e-07 and 3e-06 relative.

**And the oracle turned out not to be exact, which is the finding.** llama.cpp's
Vulkan backend evaluates a Q8_0 matmul as an *integer* dot product: it
quantises the activation to int8 in blocks of 32 first. Modelling that is
worth **233x** on this block's gate — 3.143e-03 rms down to 1.347e-05 — and
without it no tolerance against the dump means anything.

| | Exact f32 | reference's numerics | |
|---|---:|---:|---:|
| `hc_gate-0`, rms | 3.143e-03 | **1.347e-05** | **233x** |
| `hc_gate-0`, maxAbs | 6.992e-02 | 2.830e-04 | 247x |
| `hc_mixed-0`, rms | 6.329e-03 | 2.344e-05 | 270x |

## What was built

| piece | what it is |
|---|---|
| `reference/eval_dump.c` | the oracle. Links the already-built libllama, registers the same graph callback the scheduler hands every node, and writes **whole tensors** to disk. `llama-eval-callback` prints 3 values an axis and a sum; this writes all of them. |
| `llm/evaldump.go` | the reader: shape, strides, ggml type, graph order, and a `Trace` that indexes a whole pass by name. |
| `llm/model.go` | `Config` read from the checkpoint's own metadata, on top of L1's `gguf.Set`; on-demand dequantisation and a `token_embd` row gather. |
| `llm/hc.go` | `HCInit`, `HCMix`, `HCCombine` — the block, and the `Numerics` switch below. |
| `llm/hc_test.go` | every mixer and combine of the four dumped layers, each driven from llama.cpp's own value for its input. |

The dump is one model load for three stages: **143 tensors, 35.7 MB**, every
`cb()` tensor of layers 0-3 — the hyper-connections (L2), the DeltaNet
internals down to `state_predelta` and `new_state` (L3), and the QSA indexer
with its `indexer_top_k` (L4). Prompt `"The capital of France is Paris."`,
7 tokens, ids `760 6511 314 9338 369 11751 13`, recorded beside the tensors so
the Go side reproduces the same pass. It lives in `reference/out/`, which is
gitignored, so the tests skip rather than fail without it.

## The block

`build_hc_mix` and `build_hc_combine`, transcribed and then checked:

    xn      = rms_norm(res[t][c]) * gamma[c]      per stream, per token: HC
                                                  independent norms over 2560,
                                                  not one over 10240
    lo      = silu(W_down · xn / HC)              [320]
    gate    = sigmoid(W_up · lo)                  [10240]
    mixed   = mean_c (xn[c] * gate[c])            [2560]   the block's input
    inject  = W_inject · xn                       [4]      F32 in the checkpoint

    combine: res[t][c] += block_out[t] * 2*sigmoid(inject[c]/HC)

The `2*sigmoid` centres the scatter weights on 1, so a zero injection is a
plain residual add — which is how the block initialises to a no-op.

Two details that a transcription gets wrong and the dump catches. The RMSNorm
reduces over **one stream of one token**, not the concatenated 10240, because
`ggml_rms_norm` reduces over `ne[0]` and the tensor is `[2560, HC, T]`. And
the `1/HC` before the SiLU is on `lo`, the 320-vector, not on the norm.

## Against the reference

`go test ./llm/ -v`, all seven mixers and seven combines of layers 0-3, each
given llama.cpp's own value for its input so a failure is that block's:

| tensor | weights | agreement |
|---|---|---|
| `token_embd` gather, `hc_init` | Q8_0 lookup | **bit-exact**, 0 ulp |
| `hc_norm` | F32 gamma | 3.8e-08 to 8.2e-08 rms, on values to \|37.9\| |
| `hc_inject` | **F32** | 2.4e-05 to 8.3e-05 rms, on values to \|71.7\| — **3e-06 relative** |
| `hc_combine` | none (add + sigmoid) | 2.7e-10 to 1.1e-09 rms |
| `hc_gate` | **Q8_0** x2 | **6.2e-08** in 4 of 7 mixers; 6.4e-06 to 3.4e-04 in the other 3 |
| `hc_mixed` | via the gate | 4.8e-08 to 6.8e-04 |

The **6.2e-08** rows are the proof. That is 71 680 values agreeing through two
matmuls, a SiLU and a sigmoid; no wrong formula, wrong weight index or wrong
layout survives it. And the asymmetry is what located the difference in the
first place: `hc_inject` reads *the same* `xn` the gate does and agreed to
3e-06 from the start, while the gate beside it did not — because inject's
weights are F32 and the gate's are Q8_0.

## What the reference is actually computing

`ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp`, which feeds the
`pipeline_dequant_mul_mat_vec_q8_1_f32` family:

    amax over 32 values
    d     = amax / 127                 f32
    q     = round(x * (1/d))           the reciprocal, in f32
    ds    = f16vec2(d, sum*d)          the scale is stored fp16

Each of those three details is worth measuring, and they compound:

| model of the activation | `hc_gate-0` rms | |
|---|---:|---:|
| exact f32 | 3.143e-03 | |
| f16 activations, weights, or both | 3.14e-03 | **no change** — precision is not the mechanism |
| int8, blocks of 32, f32 scale | 3.529e-05 | 89x |
| int8, **fp16 stored scale** | **1.347e-05** | **233x** |

Quantising with a scale *already* rounded to fp16 — the obvious misreading —
is 4.7e-04, thirty-five times worse than quantising with the f32 reciprocal
and storing fp16. The shader is precise about which is which and so is
`llm/hc.go`.

`llm.Numerics` makes this a switch rather than a fudge: `Exact` is the
mathematical model and what the GPU port aims at; `RefQ8` is what reproduces
llama.cpp. `TestHCMixRefQ8IsTheCloserModel` asserts the ratio stays above
100x, so if the backend's path ever changes the test says so instead of
silently loosening.

### Three mixers keep a residual, and it is bounded rather than explained

Four mixers land at 6.2e-08; `blk.0.hc_attn` and `blk.2.hc_attn` at ~2e-05,
`blk.3.hc_ffn` at 3.4e-04 — with identical code, identical Q8_0 weight types
and an `hc_norm` input that agrees to 8e-08 in every one of them. The
plausible mechanism is that an int8 grid turns a last-bit difference in
accumulation order into a whole quantisation step: `lo` is only 320 values in
ten blocks, and one flipped rounding there moves all 10240 gate outputs. A
count of near-boundary `lo` values does **not** track the error cleanly (7, 4,
12, 9 against 6e-08, 6e-08, 6e-06, 3e-04), so that is a hypothesis and not a
result. What is solid is the bound: the worst mixer is 3.4e-04 rms on a gate
in [0, 1], and the tests bound rms rather than maxAbs for that reason.

## What it changes

**1. L2's acceptance criterion was wrong, and this is the replacement.**
LLM.md asks for "layer 3's output matches `llama-eval-callback` to fp16
tolerance". Downstream of a Q8_0 matmul that is unreachable *and* meaningless:
the reference is not computing in fp16, it is computing in int8, and the
disagreement is heavy-tailed, so **the gate has to be an rms bound under the
reference's own numerics**, not a maxAbs bound at fp16.

**2. The reference is already running W8A8 on its dense tensors, which is
D6's own premise.** L0d measured int8-per-token activations at 1.13x the error
of fp16 ones and called W8A8 "99% activation error"; here that is not a
projection, it is what the baseline `PPL = 4.0340` was measured on. So D3 and
D6's ~4.25-bit W4A8 bank is not giving up an activation axis the reference
keeps — **the reference already spent it**, and L8's perplexity comparison is
a like-for-like one on that axis.

**3. The fusion L2a costed is confirmed at the source level.** `inject` is
`W_inject · xn` and `hc.down` is `W_down · xn` — the *same* `xn`, 95 times a
prefill graph, one of them at 31.8 GFLOP/s. They are one matmul of 324 output
columns. `llm/hc.go` keeps them separate because it is the CPU reference and
clarity wins; the GPU kernel must not.

## What is not built yet

The fused Vulkan kernel. This is the CPU reference — the order every vertical
in this repo took, and the one L3's own task list asks for ("CPU reference
first, fp32 state, against the dump"): understand the architecture, then
debug a shader against something that already matches. The kernel L2a
specified — grouped RMSNorm, low-rank gate, 4-branch collapse, and `down`
with `inject`'s four columns fused on — is next, and it now has an oracle.

## How to reproduce

    L=/home/kube/repos/llama.cpp
    M=models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

    gcc -O2 -o /tmp/eval_dump reference/eval_dump.c \
        -I$L/include -I$L/ggml/include -L$L/build/bin \
        -lllama -lggml-base -Wl,-rpath,$L/build/bin

    /tmp/eval_dump -m $M -o reference/out/llm -c 64 \
        -p 'The capital of France is Paris.' \
        -n '^(model\.input_embed|hc_init|result_norm|result_output|hc_norm|hc_gate|hc_mixed|hc_inject|hc_combine|ple_embd|ple_gate|ple_gated_value|ple_conv_out|l_last|alpha|a_softplus|attn_gated|attn_output|attn_pregate|beta|beta_sigmoid|conv_output_raw|conv_output_silu|ffn_moe_out|ffn_out|ffn_shexp|ffn_shexp_gated|final_output|gate|gate_reshaped|gate_sigmoid|indexer_k|indexer_k_pooled|indexer_k_raw|indexer_q|indexer_score|indexer_score_tokens|indexer_top_k|k_conv|k_conv_predelta|kqv_out|linear_attn_out|linear_attn_qkv_mixed|q_conv|q_conv_predelta|shared_expert_gate|shared_expert_gate_sigmoid|state_predelta|v_conv|v_conv_predelta|z|new_state)(-[0-3])?$'

    go test ./llm/ -v        # skips cleanly without the checkpoint or the dump
