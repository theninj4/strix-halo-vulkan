<!-- LLM.md L3. Gated DeltaNet — three quarters of the layers — CPU reference,
     against llama.cpp's own activations. Cited from llm/deltanet.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L2b](l2b-hyper-connections.md) · [L2e](l2e-attention.md) · phase 1

# L3 — Gated DeltaNet, the thing the reference does instead of chunking, and a bug in the oracle

**Result: layer 0's linear attention runs in Go and matches llama.cpp on all
eighteen tensors its graph names. The recurrence's output is 3.11e-09 rms, its
786 432-value recurrent state 1.91e-08, and the layer's output 1.31e-07** —
f32 round-off, with the projections' int8 activations the only thing above it.
Layers 1 and 2, which read a residual the PLE and hyper-connection blocks have
already written into, land in the same place. **Three quarters of the model
now has a reference implementation.**

**The reference does not chunk.** `cparams.fused_gdn_ch` sends a multi-token
batch to `ggml_gated_delta_net`, whose Vulkan kernel walks the tokens **one at
a time**, one workgroup per head, 48 workgroups for the whole layer.
`build_delta_net_chunking` — 270 lines of `solve_tri`, cumulative-sum decay
masks and per-chunk matmuls — is in the tree and is not what ran. So chunking
is a **performance** question for the GPU kernel and not a correctness one,
and what it has to beat is 438 us a layer at ubatch 512.

**And the oracle has a since-fixed bug in this layer.** The build behind this
repo's trace *and* its baseline predates llama.cpp's "fix GDN normalization
from `max` to `rsqrt`". Modelling the build rather than the model is worth
**15x** on the layer — and it means L1's `PPL 4.0340` was measured with it.

**Everything above was then cross-checked against vLLM**, which implements
this architecture independently (`vllm/models/qwen4_exp/`, plus
`qwen_gdn_linear_attn.py` and a vendored flash-linear-attention). It confirms
the normalisation bug, confirms the head map *and explains why it is what it
is*, and disagrees with llama.cpp about chunking — see findings 2, 3 and 8.

| tensor | vs llama.cpp, rms | reference \|max\| | |
|---|---:|---:|---|
| `linear_attn_qkv_mixed-0` | 1.78e-06 | 31.5 | the fused [2560, 10240] projection |
| `z-0` | 1.75e-06 | 10.6 | the output gate's projection |
| `alpha-0` | 2.15e-06 | 9.33 | F32 weights — see finding 4 |
| `a_softplus-0` | 1.04e-06 | 8.77 | |
| `gate-0` | 2.09e-05 | 215.9 | 1e-07 relative; the log decay |
| `beta-0` | 2.34e-06 | 7.78 | |
| `beta_sigmoid-0` | 1.20e-07 | 1.00 | |
| `conv_output_raw-0` | 1.46e-07 | 5.25 | depthwise causal, kernel 4 |
| `conv_output_silu-0` | 1.18e-07 | 5.05 | |
| `q_conv-0`, `k_conv-0` | 1.04e-07, 6.11e-08 | 4.22, 3.49 | column ranges of it |
| `v_conv_predelta-0` | 1.36e-07 | 5.05 | |
| `q_conv_predelta-0`, `k_conv_predelta-0` | 5.08e-08, 6.01e-08 | 0.994, 0.990 | L2-normalised |
| **`attn_output-0`** | **3.11e-09** | 0.142 | the delta rule itself |
| **`new_state-0`** | **1.91e-08** | 3.75 | 786 432 values of recurrent state |
| `final_output-0` | 6.54e-08 | 1.25 | the gated RMS norm |
| **`linear_attn_out-0`** | **1.31e-07** | 1.10 | |

`state_predelta-0` is checked to be exactly zero, which is what makes every
row above a test of the recurrence rather than of the fixture. Layers 1 and 2
agree at 6.4e-10 and 5.7e-10 on `attn_output` and 2.5e-09 and 3.7e-09 on the
state; their `linear_attn_out` sits at 1.5e-05 and 1.2e-05 where layer 0's is
1.3e-07, and the whole of that gap appears at the last matmul — `final_output`
is 8.0e-08 and 9.1e-08 — so it is `ssm_out`'s int8 activation grid over a
wider input (|max| 2.0 and 5.2 against layer 0's 1.25), the heavy-tailed
behaviour L2b-3 already described. Bounded, and the same category.

## What the layer is

Per token, per head, with S the [128, 128] state, the delta rule with a forget
gate:

	S  <- S * exp(g)          decay, one scalar for the whole head
	u  <- v - kᵀS             what the state already predicts for this key
	S  <- S + k (beta·u)ᵀ     write the correction back
	o  <- qᵀS / sqrt(128)     read it out

Everything in front of that is shaping — one fused [2560, 10240] projection
for q, k and v end to end, a depthwise causal convolution of kernel 4, a SiLU,
an L2 normalisation of q and k, and two [2560, 48] F32 projections for the
decay and the write strength — and everything behind it is a gated RMS norm
against a second [2560, 6144] projection and the output matmul. 36 of the 48
layers, 2.09 B parameters, and **1.6% of llama.cpp's prefill graph** (L2a-1).

## Finding 1 — prefill is a sequential scan, not a chunked matmul

The evidence is in the dump's shapes rather than in the source.
`q_conv_predelta-0` is **[128, 16, 7]** — sixteen heads. The non-fused path
(`build_delta_net_chunking`) begins by `ggml_repeat_4d`-ing q and k up to the
48 value heads, so under it that tensor would have been [128, 48, 7]. It is
not, so `cparams.fused_gdn_ch` is set and `ggml_gated_delta_net` ran.

That op's Vulkan kernel (`gated_delta_net.comp`, 189 lines) is the recurrence
transcribed directly: one workgroup per (head, sequence), `COLS_PER_WG`
columns of the state held in registers across the whole token loop, a subgroup
reduction for each of the two dot products, and `for (uint t = 0; t <
n_tokens; t++)` around all of it. There is no matmul anywhere in it.

**So the CPU reference is the sequential recurrence**, and the chunked form
stays a performance option rather than a specification. What it would have to
beat, from `results/l2a_prefill_ops.csv`:

| | per layer | total | % of graph |
|---|---:|---:|---:|
| `GATED_DELTA_NET`, ubatch 512 | 438 us | 15.8 ms | 1.29% |
| `GATED_DELTA_NET`, ubatch 2048 | 2058 us | 74.1 ms | 1.63% |
| `SSM_CONV_SILU`, ubatch 512 | 188 us | 6.8 ms | 0.56% |
| `L2_NORM`, ubatch 512 | 63 us | 4.5 ms | 0.37% |

438 us for 48 workgroups running 512 dependent iterations each. The chunked
form trades that serial chain for a triangular solve on a 64x64 system plus
four matmuls per chunk per head — far more arithmetic, all of it parallel.
Which wins on this hardware is L3's kernel question and is **not** settled
here.

## Finding 2 — the head map is modulo, and ggml's own two broadcasts disagree

48 value heads read 16 key heads, and there are two ways ggml expresses that:

- `ggml_repeat` **tiles**: head h reads source head `h % 16`.
- `ggml_mul_mat`'s batch broadcast **divides**: `i03 = i13 / r3`, so head h
  reads source head `h / 3`.

The chunking path takes the second, because it reshapes q and k into the
matmul's batch axis. The fused op takes the first — `iq1 = iv1 % neq1` in
`ggml_compute_forward_gated_delta_net_one_chunk`, `head_id % neq1` in the
shader. **They agree on 16 of the 48 heads**, which is exactly enough for the
wrong one to look nearly right, and what it produces is a real tensor built
from the wrong heads. `TestDeltaNetHeadMapIsModulo` runs the same recurrence
both ways:

| head map | `attn_output-0` rms | maxRel |
|---|---:|---:|
| **modulo, `h % 16`** | **3.11e-09** | 4.9e-06 |
| divide, `h / 3` | 2.71e-03 | 9.44 |

**870 000x** — the same category of negative control as L2c's
`TestHCGPUUnpermutedUpIsWrong`.

### …and vLLM divides, because this checkpoint's V heads are permuted

vLLM maps v-head h to k-head **`i_h // (H // Hg)`** — `h / 3`, the *other*
map — in flash-linear-attention's `chunk_fwd_kernel_o` and
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`. Two mature implementations,
opposite maps, and both produce a working model. The resolution is that they
are not reading the same bytes:

> *"Linear attention may has num_k_heads < num_v_heads. The HF weights store V
> heads grouped by K head: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...]. ggml binary
> ops use tiled broadcast: [K0, K1, ..., K0, K1, ...]. We reorder V heads to
> tiled order so ggml_repeat can replace the expensive interleaved repeat."*
> — `_LinearAttentionVReorderBase`, llama.cpp `conversion/qwen.py`

The converter transposes the (16, 3) head grid, so GGUF v-head `r*16 + g` is
HF v-head `g*3 + r` and `h % 16` recovers `g`. It does this to **seven tensor
families at once** — `in_proj_qkv`'s V rows, `in_proj_z`, `in_proj_a`,
`in_proj_b`, `A_log`, `dt_bias`, `conv1d`'s V channels and `out_proj`'s
columns — because every per-v-head tensor has to move together.

**So `h % 16` is a fact about the GGUF, not about the architecture**, and
`DeltaNetTrace.NewState`'s head order is the GGUF's rather than the model's.
That is only trivia while phase 1 reads UD-Q4_K_XL. It stops being trivia at
**L8**, whose stated alternative is to re-quantise "from the 360 GB bf16
(clean — and unsloth publish their imatrix)": that path reads HF's ordering,
where `h % 16` is wrong on 32 of the 48 heads, and it would produce a model
that is subtly wrong in **36 of the 48 layers** with nothing but perplexity to
notice. Whatever L8 does, it either replicates `_reorder_v_heads` across all
seven families or switches `kHeadOfV` to divide — and the two must not be
mixed.

## Finding 3 — the oracle's build has a GDN normalisation bug, and it is worth 15x

`build_gdn_l2_norm` normalises q and k over the head. There are two spellings
of it in llama.cpp's history and they are one commit apart:

- **`ggml_l2_norm(x, eps)`**, which divides by `max(|x|, eps)`;
- **`ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))`**, which divides by
  `sqrt(|x|² + eps)`.

Commit 5fdfa6282, *"models : fix GDN normalization from `max` to `rsqrt`"*
(#28068), replaced the first with the second. This repo's llama.cpp build is
**cff184438, which predates it** — and `cff184438` is not an incidental
detail: it is the build that produced the trace every test here compares
against, `llama-bench`'s **pp2048 388.60 / tg128 25.15**, and
**`PPL = 4.0340 ± 0.02283`**.

At |x| near 1 the two differ by about eps/2 = 5e-07 relative, so the fix
changes nothing anyone would notice in a generation. It is 15x the noise floor
of this comparison:

| | `k_conv_predelta-0` | `attn_output-0` | `linear_attn_out-0` |
|---|---:|---:|---:|
| **max (the build)** | **6.01e-08** | **3.11e-09** | **1.31e-07** |
| rsqrt (the model) | 9.16e-07 | 1.24e-07 | 1.91e-06 |
| | 15.2x | 40x | 14.6x |

**vLLM settles which one is the model.** Its fused post-conv Triton kernel
does `q_inv = 1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS)` with `L2NORM_EPS = 1e-6`
— the rsqrt form, independently of llama.cpp — so #28068 is a fix and not a
change of convention, and our default is pointed the right way.

So `DeltaNetConfig.QKNorm` selects the spelling, `L2Rsqrt` is the default
because it is the model, and the tests set `L2Max` because that is the
build. `TestDeltaNetL2NormIsTheBuilds` asserts the preference, so **the day
llama.cpp is rebuilt past 5fdfa6282 the test fails and says why** rather than
the residual quietly growing by 15x across L4 to L7.

Two consequences worth carrying forward. First, the dump and the perplexity
baseline are **the same build**, so L8's comparison against 4.0340 stays
like-for-like as long as nothing is rebuilt — and stops being so if anything
is. Second, this is the first time the oracle has been wrong about the
*model* rather than merely imprecise about the arithmetic; L2b's int8
activations, L2e's fp16 cache and L2e's fp16 F32 matmul are all faithful
evaluations of the right formula, and this is not.

## Finding 4 — the oracle's fp16 matmul has a threshold, and it is 8 columns

L2e-3 found that **a matmul between two F32 tensors is evaluated on the fp16
matrix cores**, worth 69x on the indexer's score, and left it stated as a
property of the backend. `ssm_alpha` and `ssm_beta` are F32 [2560, 48] weights
read by an F32 activation — the same situation — so the obvious move was to
model them in fp16 too. Measured on `alpha-0`:

| | rms |
|---|---:|
| **f32 throughout** | **2.15e-06** |
| both operands rounded to fp16 | 1.84e-04 |

**86x the other way.** These matmuls are f32, and L2e-3 as written does not
generalise.

The reason is a dispatch threshold rather than a type rule. `ggml_vk_mul_mat`
sends a matmul to `ggml_vk_mul_mat_vec_q_f16` — which keeps f32 operands —
whenever the output has at most `mul_mat_vec_max_cols = 8` columns, and to the
coopmat GEMM otherwise. The dump's prompt is **7 tokens**, so every per-token
projection in it takes the vector path; the indexer's score has 4 heads x 7
tokens = **28** columns and takes the GEMM path. Both observations fall out of
one line.

**This is a prediction, and L6 has to re-check it**: at a real 512-token
ubatch every projection in the model crosses to the GEMM path, so L2e-3's fp16
arithmetic applies *there* and not here. **A tolerance measured against a
7-token dump is not automatically a tolerance at 512**, and that is now true
of L2b, L2d, L2e and this stage alike.

## Finding 5 — softplus flushes to zero, and being more accurate is being wrong

ggml's `op_softplus` is `(x > 20) ? x : logf(1.0f + expf(x))`, and the add is
in **float32** — so any x below about -16.6 gives `1.0f + tiny == 1.0f` and
the result is **exactly zero**. One of `a_softplus-0`'s 336 values is exactly
zero for that reason. `math.Log1p(math.Exp(x))`, the numerically better
spelling, returns 2e-09 there and disagrees with the reference. The Go side
does it ggml's way, f32 add included.

It changes nothing downstream — a decay of `exp(0)` against `exp(-1e-08)` —
but it is the same class of thing as L2b's activation quantiser: the
reference's arithmetic is the specification, not the mathematics it
approximates.

## Finding 6 — the state is stored transposed, and that is the right layout

ggml's `new_state` has **ne[0] = i, the key axis, and ne[1] = j, the value
axis**, holding S[i][j]. So a row of the stored tensor is a *column* of S —
and a column of S is exactly what all three of a token's operations touch:
`kᵀS` reads column j, the outer-product update writes column j, `qᵀS` reads
column j. Every access in the recurrence is contiguous, and the two dot
products and the update over one column can share one register-resident shard.

That is not a convenience of the reference's implementation; it is why its
kernel can hold the state in registers across the whole token loop, and this
file keeps the layout for the port's sake rather than transposing into the
mathematically tidier form.

## Finding 7 — two things cross a batch boundary, not one

`TestDeltaNetChunkBoundary` runs the 7-token prompt as 3 + 4 and compares
against the single pass:

| | |
|---|---|
| `linear_attn_out`, 3+4 against 7 | **bit-identical**, maxAbs 0 |
| `new_state`, 3+4 against 7 | **bit-identical**, maxAbs 0 |
| second chunk with no carried state | 4.48e-02 rms — the control |

The second row is the one the task list asked for. The **first** row is the
one that is easy to get wrong, because the recurrent state is not the only
thing that carries: the depthwise convolution reaches three tokens back, so
`DeltaNetState` holds its last three columns as well, in the reference's
layout (token axis fastest, which is what `ggml_ssm_conv` slides over). A
kernel that carries the [128, 128, 48] state and forgets the 120 KB
convolution window is wrong in a way nothing but this test sees — and at L7,
where every decoded token is its own batch, it would be wrong on every token.

## Finding 8 — the two mature implementations disagree about chunking

Finding 1 established that llama.cpp runs a sequential scan for prefill and
declines its own chunked path. **vLLM does the opposite**, and the split is
along the batch shape rather than the hardware:

| | prefill | decode |
|---|---|---|
| llama.cpp | `ggml_gated_delta_net`, sequential scan | same kernel, one token |
| vLLM | **`chunk_gated_delta_rule`**, chunk size 64 | `fused_sigmoid_gating_delta_rule_update` |

So the question L3b has to answer is a real open question that two teams have
answered differently, not an oversight in one of them. And flash-linear-attention
gives the chunked form as a **four-stage decomposition** that is worth having
written down before designing a kernel:

1. `chunk_scaled_dot_kkt_fwd` — the decayed `K β Kᵀ` matrix per chunk;
2. `solve_tril` — invert `(I + tril(A))` on a 64x64 unit-triangular system;
3. `chunk_delta_h` — carry the state across chunks (the only serial stage,
   and it is serial in *chunks* rather than in tokens: 8 steps at 512 tokens
   where the scan has 512);
4. `chunk_fwd_o` — the intra-chunk output, `q h` plus the causal `q kᵀ v_new`.

Stages 1, 2 and 4 are matmuls on the shapes this repo's WMMA kernels already
run, and stage 3 is where the serial dependency goes. That is the concrete
form of §3.6's "find out whether it is matmul-bound or serialised on the state
update" — both, in different stages, and the chunk size is the dial between
them. Nothing here is measured on this hardware; it is the blueprint L3b tests
against the 438 us the scan costs.

## One fix to the oracle reader

Nine of the trace's 143 tensors are **non-contiguous ggml views** — `q_conv`,
`k_conv` and `v_conv_predelta` of the three linear layers, which are column
ranges of the convolution's output and so carry its 40 960-byte row stride.
`eval_dump.c` copies `ggml_nbytes(t)` bytes from the tensor's own pointer,
which for a view is **the region of the parent it spans**: 253 952 bytes where
the tensor has 57 344 of values. Read flat, `q_conv-0` came back at 1.61e-01
rms — uncorrelated — while `q_conv_predelta-0`, the normalisation of that very
tensor two lines later, was already at 1.3e-07.

The strides are in the file header, so `ReadDump` now gathers with them.
After: **1.04e-07**. No other tensor in the trace is affected (every other
dump is contiguous), and `eval_dump.c` did not have to change.

## The second implementation

`../vllm` (`vllm/models/qwen4_exp/`, `vllm/model_executor/layers/mamba/gdn/
qwen_gdn_linear_attn.py`, `vllm/third_party/flash_linear_attention/`) is an
independent implementation of this architecture, and it earned its place in
this write-up three times over: it confirmed the normalisation bug, it
explained the head map, and it disagreed about chunking. What it **cannot**
be is a second oracle — it runs bf16 Triton on CUDA against our Q4_K_XL on
Vulkan, and no tensor-level comparison exists without running a 180 B model
here. Its role is **arbiter of formulas, not of numbers**, which is exactly
where llama.cpp has now been unreliable twice in this one layer.

Four things it corroborates without argument, each of which was read out of
llama.cpp alone until now: the packed layout is `[q | k | v]` with z, a and b
separate (`gqa_interleaved_layout=False` — the interleaved variant is
Qwen3-Next, a different model); `g = -exp(A_log) * softplus(a + dt_bias)`;
`beta = sigmoid(b)`; and the softplus cutoff is 20. Its softplus also
**flushes to zero in f32** the same way ggml's does, so finding 5 is a
property of the arithmetic rather than a llama.cpp quirk.

## How to reproduce

    go test ./llm/ -v -run TestDeltaNet

Layers 0, 1 and 2 are the linear-attention layers the trace covers — layer 3
is full attention, which is L2e's — and `hc_mixed-N` is llama.cpp's own value
for each layer's input, so every comparison is that layer's own rather than an
accumulation.

## What is not built

**The GPU kernel**, which is the rest of L3, and finding 8 says the first
question is a shape question rather than a scheduling one: llama.cpp's
sequential scan is 48 workgroups deep in a 512-iteration serial loop, vLLM's
chunked path is a triangular solve plus three matmul stages per 64-token
chunk, and the two implementations disagree. Neither is obviously right on
this hardware and §3.6 predicted neither. The one thing already known is that
the state layout above wants to stay in registers, which is what makes the
serial form viable at all — and 48 workgroups against this GPU's occupancy is
what makes it suspect.

**Decode.** Everything here is prefill from position zero. The state carries
correctly across a batch boundary, which is the piece L7 needs, but the
1-token path has its own kernel in the reference (`LLM_FUSED_OP_GDN_AR`) and
its own problem: 48 workgroups, one token, 3.1 MB of state read and written
per layer. That is **113 MB of state traffic a token across the 36 layers**
against a 6.334 GB weight budget — 1.8%, so not a bandwidth problem, but
`results/l2a_prefill_ops.csv` prices the reference's decode `GATED_DELTA_NET`
at **392 us, 0.95% of a step**, on a step L2a-5 says is already losing 22% to
dispatch count.
