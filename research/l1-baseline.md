<!-- LLM.md L1. The checkpoint, the reader, and the number to beat. Cited from
     gguf/, cmd/gguf, cmd/llm and bench/modelshapes.go. -->

[← LLM.md](llm-vertical.md) · [research index](README.md) · [L0a](l0a-bank-range.md) · [§5.1](5.1-memory-types.md) · phase 1

# L1 — the checkpoint, and llama.cpp's number

**Result: the bar is `pp2048 388.60 ± 1.89` and `tg128 25.15 ± 0.02` tok/s**,
and the Go side can now read the checkpoint that produced them — a GGUF
reader checked tensor-for-tensor against the Python inventory, five dequant
paths **bit-exact against llama.cpp's own `to_float`**, and a tokenizer
**exact on 213 of 213 tokens against `llama-tokenize`**.

Two things in LLM.md turned out to be wrong, and both matter for what comes
next:

1. **Prefill is not where the arithmetic said it was.** §3.4 and §2.2 were
   added up into "~2000 tok/s prefill at a 2048 chunk". llama.cpp gets
   **388.60**, and it does not improve past that: 8192 gives 392.95. Either
   the model has a lot of prefill in it that is not matmul — DeltaNet, the
   QSA indexer, 194 hyper-connection projections a token — or llama.cpp's
   Vulkan path leaves 5x on the table. Both are worth knowing and only one
   of them is good news.
2. **`bench/modelshapes.go` was wrong in four rows and missing four
   families.** The real tensor table has `attn.q` at 12288 (not 6144), K and
   V as two separate 512-wide matrices, a *single fused* 10240-wide DeltaNet
   input projection (not a 2048/6144 pair), and no rows at all for the
   hyper-connections, the QSA indexer or the PLE block. The corrected table
   reads **6.665 B weights a token against the checkpoint's own 6.671 B**,
   0.09% apart, where the old one had 200 matmuls a token unaccounted for.

## The baseline

`llama-bench`, build `cff184438`, Vulkan on RADV STRIX_HALO, the model as
shipped (`UD-Q4_K_XL`, 103.68 GiB, 176.94 B params), default `-ngl -1`,
mmap on:

| test | tok/s | run 2 | spread |
|---|---:|---:|---:|
| pp512 | 315.21 ± 0.03 | 313.62 ± 1.74 | 0.5% |
| pp2048 | — | **388.60 ± 1.89** | |
| pp8192 | — | 392.95 ± 1.04 | |
| tg32 | — | 25.12 ± 0.03 | |
| tg128 | 24.96 ± 0.03 | **25.15 ± 0.02** | 0.8% |

Both runs are two repetitions each (`-r 2`), run separately, nothing else on
the GPU — the two-run convention `TODO.md` asks for. A real generation agrees
with the bench: `llama-completion -n 64 --temp 0` reports **24.66 tok/s**
eval and produces coherent text.

### The accuracy reference

`llama-perplexity -f models/wikitext-2-raw/wiki.test.raw -c 2048 -b 2048`,
145 chunks, 6.62 s a pass, 15.2 minutes:

    Final estimate: PPL = 4.0340 +/- 0.02283

That is **the number phase 2 has to stay near**. L8 re-quantises the same
weights to ~4.25 bits on everything streamed, and the honest test of D3 is
this figure re-measured on our own bank with the same corpus, the same context
and the same chunking — none of which can change without the comparison
losing its meaning, which is why all three are written down here.

### What the decode number means

The inventory says a token reads **6.334 GB**. At 25.15 tok/s that is
**159.3 GB/s**, which is **65.8% of the 242 GB/s** §1.7 measures on a real
decode GEMV and 67.5% of §0.4's 236 GB/s peak. So llama.cpp is leaving a
third of the bus unused on a workload that is nothing but bus, and the 38.2
tok/s ceiling this vertical was scoped against is real but not what the
reference implementation reaches.

That is the honest shape of the target. **Phase 1's job is 38.2, not 25.15**,
and phase 2's ~67 is then 2.7x the reference rather than 1.8x.

### Residency: llama.cpp does keep it resident, and it already does D2

An open question in LLM.md was whether llama.cpp holds 82 GB device-local or
pages it. Measured with `free` during a run: **77 GiB resident**, steady,
against the 82.52 GB (76.86 GiB) the inventory calls the resident core. It
matches to within the rounding.

And the reason it matches is that llama.cpp already does **D2**:
`src/models/qwen4exp.cpp` creates `per_layer_token_embd` with
`TENSOR_READ_LAZY`, which the loader documents as *"read rows on demand
instead of loading whole tensor; requires mmap"*. The 28.80 GB n-gram table
is gathered from the mapping, host-side, exactly as D2 proposes. So D2 is not
a deviation from the reference implementation; it is what the reference
implementation does, and the baseline above is a like-for-like comparison.

## What was built

| piece | what it is | checked against |
|---|---|---|
| `gguf/gguf.go` | mmap'd GGUF reader: the value grammar, the tensor table, `general.alignment`, and the split convention (`split.no`/`split.count`/`split.tensors.count`), openable from any shard | `cmd/gguf -check tensors.json`: **1224 of 1224 tensors, 0 disagreements** against `reference/gguf_inventory.py` |
| `gguf/dequant.go` | Q4_K, Q5_K, Q5_1, Q8_0, IQ4_NL, F32, F16, BF16 | `reference/dequant_ref.c` through `ggml_get_type_traits(t)->to_float`: **bit-exact**, ulp for ulp, on 2944 values |
| `cmd/gguf` | the Python inventory's Go counterpart — groups, quant mix, decode budget | reproduces LLM.md's table exactly (82.52 GB core, 6.334 GB/token, 38.2 tok/s, 76% dense) in **47 ms** |
| `zimage/tokenizer` + `cmd/llm` | `FromVocab` builds the BPE from the GGUF's metadata arrays; `split()` gained the `qwen35` pre-tokenizer | `llama-tokenize --ids`: **213 of 213 identical** |
| `bench/modelshapes.go` | the qwen3.8-flash-next rows, rewritten from the tensor table | 6.665 B vs the checkpoint's 6.671 B weights a token |

### What the corrected table changes

`go run ./cmd/bench shapes`, re-run against it (313 rows, `results/shapes.csv`):

| | before | after |
|---|---:|---:|
| rows in the table | 23 | 39 |
| decode weights a token | 5.779 B — **13.4% short** | **6.665 B** |
| decode matmuls a token | 1861 | **2081** |

At 4-bit weights the summary now prices a decode step at **3.33 GB, 13.8 ms,
73 tok/s before attention**, with the 2081 dispatches adding **0.62 ms of
launch cost — 5%** (§4.1). That 5% is new information and it is a phase-2
constraint: the 220 matmuls the table was missing are the hyper-connections,
which are 194 of them, narrow (320 wide), and unavoidable. A decode step that
means to reach ~67 tok/s has ~15 ms to spend, so **launch cost is 4% of the
budget before a single weight is read**, and batching the hyper-connection
projections is worth more than it looks.

### The two pre-tokenizers

`tokenizer.ggml.pre` is `qwen35`, and llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35`
differs from `QWEN2` in exactly one character class, twice:

    qwen2   …|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|…
    qwen35  …|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|…

A combining mark joins the letters before it instead of being taken as a
symbol. `e` + U+0301 is one piece under qwen35 and two under qwen2 — which is
why the test corpus carries a decomposed "é", a bare combining acute, and a
Devanagari cluster beside the ordinary English. The corpus also covers the
places byte-level BPE usually breaks: contractions, digits (which split one at
a time), a tab and a blank line, CJK with no spaces, a ZWJ emoji sequence, and
the chat and multimodal specials, which have to be matched whole.

### Why the dequant oracle is a C program and not a Python one

The five formats are transcriptions of `ggml-quants.c`. A test that restated
them would be checking the transcription against itself, and a numpy
reimplementation would only move the question. `reference/dequant_ref.c` links
against the already-built `libggml-base.so` and calls the *same function*
every llama.cpp CPU path calls, over bytes the Go test generates. Comparison
is bit-exact rather than within a tolerance: one ulp of disagreement is a real
disagreement about the format.

The inputs are pseudorandom with one constraint — every fp16 scale field is
forced to an ordinary magnitude, because a random 16-bit pattern is Inf or NaN
one time in 32 and a NaN scale makes a strict comparison meaningless. Nibbles,
sign bits, the packed 6-bit K-quant scales and the fifth-bit planes stay fully
random, which is the part that would catch a layout error.

## The download

114 GB in **18 minutes** at 91-106 MiB/s — four `UD-Q4_K_XL` shards and the
2.79 GB MTP head, by `reference/fetch_llm_checkpoint.sh`, which is
resumable and checks each file's size against the HF API's before moving on.
Shard 1 is 11 MB and holds **no tensors at all**: it is the metadata shard —
67 keys, including the 248 320-entry vocabulary, the merge table and the chat
template. The tensors live in shards 2-4 (297, 752, 175).

## How to reproduce

    # the inventory, Go and Python, and the check between them
    go run ./cmd/gguf models/Qwen3.8-Flash-Next-GGUF/…-00001-of-00004.gguf
    python3 reference/gguf_inventory.py s1.gguf s2.hdr s3.hdr s4.hdr   # writes tensors.json
    go run ./cmd/gguf -check tensors.json …-00001-of-00004.gguf

    # the dequant oracle (only needed to regenerate the committed testdata)
    L=/home/kube/repos/llama.cpp
    gcc -O2 -o /tmp/dequant_ref reference/dequant_ref.c \
        -I$L/ggml/include -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin
    go test ./gguf/ -run TestDequantMatchesGGML -updatedequant
    /tmp/dequant_ref gguf/testdata/dequant_in.bin gguf/testdata/dequant_ref.bin

    # the tokenizer, against llama.cpp
    $L/build/bin/llama-tokenize -m …-00001-of-00004.gguf \
        -f cmd/llm/testdata/tokenizer_corpus.txt --ids
    go run ./cmd/llm -tokenize-file cmd/llm/testdata/tokenizer_corpus.txt -ids-only

    # the baseline
    $L/build/bin/llama-bench -m …-00001-of-00004.gguf -p 512,2048,8192 -n 32,128 -r 2
