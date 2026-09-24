# LLM-VISION — image input for the text vertical

> **Live tracking and session handoff doc, opened 2026-09-24.** Stage letters
> are **V**. When the vertical closes, this file is frozen to
> `research/llm-vision.md` like the others and `TODO.md` gets the one-line
> summary. Until then: tick a stage when its gate passes, put its measured
> numbers under it, and keep **§ Handoff** at the bottom current. A new session
> should be able to start from there.

## What we are building

`qwen3.8-flash-next` is a vision-language checkpoint (`Qwen4ExpForConditionalGeneration`,
card: *"Causal Language Model with Vision Encoder"*). We serve its text half
only, and every API door **refuses** an image block with a reason
(`api/chat.go:152`, `api/messages.go:456/489`, `llm/chat.go:23`). The goal:
image content parts (`image_url` on chat, `input_image` on responses, `image`
on messages) answered by the model. Video stays refused for now (V11).

## What the model is (V0: established 2026-09-24)

Three sources were read against each other. **HF transformers is the ground truth**
(`.venv/…/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, 5.17.0). llama.cpp
(`~/repos/llama.cpp`, mtmd) is the only thing that can *run* the whole model,
and it is our oracle for the LLM, as it was from L2 on. vLLM (`~/repos/vllm/vllm/models/qwen4_exp/`)
is the third opinion.

**The tower: the same ViT `qimage/vision` already runs.** The ViT is `mmproj-BF16.gguf`
(0.91 GB, 449 M params, now in `models/Qwen3.8-Flash-Next-GGUF/`), `clip.projector_type
qwen3vl_merger`. It is Qwen3-VL's: 27 blocks, hidden 1152, 16 heads × 72, FFN 4304
gelu-tanh, patch 16, temporal patch 2, merge 2×2, 48×48 learned positions,
image mean/std 0.5, LN eps 1e-6. That is `qimage/vision` (IMAGE.md Q8), gated and on the
GPU for Qwen-Image-2.1's edit encoder, with its five traps already
written down (block-major token order, interpolated position grid, two GELUs,
two mergers, axial NeoX rope). What differs:

| | Qwen-Image-2.1's tower | this model's tower |
|---|---|---|
| merger output | 4096 | **2560** (the LLM's `n_embd`) |
| deepstack | 3 mergers at blocks 8/16/24 | **none** (`deepstack_visual_indexes: []`) |
| weights | safetensors, `model.visual.*` | GGUF `v.*`/`mm.*`, or HF shard 1 of 131 |
| patch embed | one Conv3d `[1152,3,2,16,16]` | GGUF splits it by time: `v.patch_embd.weight` + `.weight.1`, each `[16,16,3,1152]` |

GGUF name map (the output merger is the pre-shuffle kind, norm 1152 wide):
`v.blk.N.{ln1,ln2,attn_qkv,attn_out,ffn_up,ffn_down}` → `blocks.N.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`;
`v.post_ln` → `merger.norm`; `mm.0`/`mm.2` → `merger.linear_fc1`/`linear_fc2`;
`v.position_embd` → `pos_embed`.

**The prompt.** The template renders an image part as
`<|vision_start|><|image_pad|><|vision_end|>`, optionally prefixed `Picture N: `
(`add_vision_id`). A system message with an image raises. The processor then expands
`<|image_pad|>` (id **248056**) to one token per merged patch. Other ids:
vision_start 248053, vision_end 248054, video_pad 248057.

**Preprocessing** (`preprocessor_config.json`): `Qwen2VLImageProcessor` on
the torchvision backend. Smart-resize to multiples of 32, `shortest_edge` 65 536 px
(64 tokens) and `longest_edge` 16 777 216 px (**16 384 tokens, 65 536
patches**). **Alpha is dropped, not composited** (V2 measured it). Then
torch's native uint8 antialiased bicubic, then `(x − 127.5) / 127.5`. That is
*not* the Pillow Lanczos `qimage/pipeline/resize.go` ports, and *not* Pillow's
BICUBIC either (±1 on ~0.5% of samples). V3 has the details.

**Positions: 3-D interleaved M-RoPE, and position ≠ cell.** `get_rope_index`
(HF) and `mtmd_image_tokens_get_decoder_pos` (llama.cpp) agree. An image starting
at text position `p` with merged grid `gh × gw` gives its token `(r, c)` the
position `(t, h, w) = (p, p + r, p + c)`. The next text token is at `p + max(gh, gw)`, **not**
`p + gh·gw`. So after an image, a cell's rotary position is its cell index
minus a per-sequence delta (HF's `mrope_position_deltas`). Section rule
(`[11, 11, 10]` over 32 rotary pairs, interleaved): pair `j` reads t when `j%3==0`,
h when 1, w when 2. `llm.RoPEMulti` already implements the rule and feeds it
`(pos,pos,pos)`. On text that is NeoX, which is why nothing has needed a
position table yet (L2e).

- **vLLM disagrees and is overruled.** Both of its `get_mrope_input_positions`
  return a plain `arange` for every token. Two of three sources, including the
  authors' own `transformers` port, say 3-D.
- **The causal mask stays by cell.** HF passes 3-row position ids, so
  `text_position_ids` is None and the mask is by cache index. llama.cpp masks
  by position, then raster `(y, x)` within one `t`, which is the same order.
  So no mask change: the image's tokens are causal in raster order.

**PLE at image slots: we follow HF, and so does current llama.cpp.** HF
hashes `ple_input_ids = input_ids`: the `<|image_pad|>` id at every image
slot, with ordinary n-gram history. So an image token's trigram is (pad, pad,
pad), and `<|vision_end|>`'s is (end, pad, pad). *(Corrected at V6: a stale
comment in llama.cpp's `llama-kv-cells.h` says image cells store a null token,
but `llama-kv-cache.cpp` stores `ple.image_token_id` in them, and its
`get_prev_tokens` then yields HF's trigrams. The oracle needs no patch.)*

**No new LLM kernels are needed** beyond positions. The image rows enter where
token embeddings do. `Graph.appendN` already gathers embeddings on the host
and uploads them (`g.hc.UploadInit`), so an image is rows substituted into that
upload.

## Budget

- Weights: +0.91 GB (BF16 → staged fp16) on the LLM machine, which runs ~98 GB
  today (P18). Tower activations at the 16 384-token cap are 65 536 patches ×
  1152 fp16 ≈ 150 MB a buffer, and the attention is **full** (no windows), so
  its cost is quadratic in patches.
- KV: an image is `gh·gw` cells. A 1024×1024 image is 32×32 = 1 024 cells, and one at
  the processor's cap is 16 384. Positions consumed are only `max(gh, gw)`, which is
  immaterial against 262 144 cells.

## Stages

- [x] **V0 — ground truth and decisions.** Above. Files fetched:
  `models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf`, and
  `models/Qwen3.8-Flash-Next-HF-vision/` (configs, tokenizer, template and
  `model-00001-of-00131.safetensors`, which holds all 333 `model.visual.*`
  tensors, 1.04 GB). The HF shard is what makes a Python fp32 oracle possible without the
  180 B model.

- [x] **V1 — the tower loads from the mmproj.** `qimage/vision/gguf.go`,
  `LoadGGUF(path, blocks)`: config from `clip.vision.*`, the two per-frame
  patch kernels interleaved per channel into Patchify's `[1152, 1536]` row
  layout, pre-shuffle merger `v.post_ln`/`mm.0`/`mm.2` to 2560, and a refusal
  if the file has deepstack. **Gate `TestGGUFMatchesSafetensors`: all 333
  tensors bit-identical** to HF's `model.visual.*`, and the implied Config
  equals `config.json`'s. The gate discriminates: the two time frames of the
  patch kernel differ by up to 0.0065 (max |w| 0.075), so a concatenation
  instead of an interleave fails it. Q1 settled for now: the package **stays
  in `qimage/vision`** and `llm` imports it. A move to `vision/` is a rename to do
  when nothing else is editing qimage.

- [x] **V2 — the fp32 oracle dump.** `reference/dump_llm_vision.py` →
  `reference/out/llmvision/` (9 s, 36 tensors). HF's processor and
  `Qwen4ExpVisionModel` from shard 1 over four generated cards: `sq` 256²
  RGB (no resize), `odd` 517×300 (a downscale on both axes → 18×32 patches),
  `tiny` 150×100 (an upscale → 14×20), and `rgba` (sq with a gradient
  alpha). Plus the square card in float64, a two-image chat prompt (272
  tokens, 208 pads; `get_rope_index`'s positions end at 87, **delta −184**),
  HF's own PLE n-gram ids for that prompt (the module built on the meta device with
  its lookup stubbed), and a `smart_resize` table.
  **Gate `TestLLMTower` (CPU, 3.5 min): merged rows rel 5.7e-5 (sq) and 4.4e-5
  (odd) against HF fp32, and 1.3e-5 against float64**. The Go tower is nearer the
  truth than the reference, because it accumulates in float64. HF fp32 vs float64 is
  3e-6 absolute on the merged rows.
  **Gate `TestPLEImagePrompt`: `llm.PLERows` over HF's ids is identical to HF's
  hash, 272 × 16 rows.** So the graph needs no PLE change for images: keep
  248056 in `ids`, and llama.cpp's cut-at-image behaviour is the one that
  differs.

- [x] **V3 — preprocessing in Go.** `llm/pixels`: `Decode`, `FromImage`,
  `Processor.SmartResize`, `Resize` and `Planes`, whose normalized `[3, H, W]`
  output goes to `vision.Config.Patchify`.
  - **Resize = torch's uint8 AA kernel.** On AVX2/AVX512 CPUs torchvision takes
    torch's native uint8 path for bicubic. That path is Pillow-SIMD's resampler:
    Pillow's coefficient plan and cubic a = −0.5, but weights quantized to int16
    at a per-axis precision (the largest keeping 2·max-weight under 2^15), where
    Pillow uses a fixed 22 bits. Horizontal pass first, uint8 between passes, an
    unchanged axis skipped. Pillow's own BICUBIC is off by one on ~0.5% of samples.
    `smart_resize` uses Python's half-to-even `round()`.
  - **Gates `TestSmartResize` (17 sizes, every branch) and `TestPixelValues`
    (four cards: no resize, downscale, upscale, alpha): bit-identical.**
  - **Decoding.** PNG, JPEG and GIF from the standard library. **This module has
    no dependencies, so WebP is refused with `ErrUnsupported` (Q6).** EXIF
    orientation is not applied, because HF doesn't apply it. **Gate `TestDecode`:
    PNG (RGBA, palette with a transparent entry, grey) and GIF are bit-identical
    to PIL's `convert("RGB")`.** Two places needed care. Go's generic conversion
    premultiplies, which turns a transparent palette colour black and rounds
    16-bit, so `FromImage` reads those types directly. Go's GIF decoder
    overwrites the transparent entry, so `restoreTransparent` reads it back from
    the file's colour table.
  - **JPEG is measured, not gated bit-exact.** On Go's planes, `jpeg.go` applies
    libjpeg-turbo's fancy chroma upsampling (h2v1, h1v2, h2v2 with libjpeg's
    biases and edge replication) and jdcolor.c's fixed-point colour conversion.
    That takes the gap to PIL from **mean 6.0 levels, max 74** (synthetic
    card) and 0.1–0.5 mean (photos) to **mean 0.02–0.03, max 3, on ~2% of
    samples**. What remains is Go 1.26's IDCT, a fixed-point Loeffler transform
    that is not jidctint's.
  - **Is that residual worth a decoder of our own? No.** `TestLLMJPEGGap` runs
    four JPEGs (a card and three photos, up to 2592×1754) decoded by PIL and by us
    through the device tower. The decode gap moves the merged rows by 0.018–0.106
    RMS. **The control, ±1 level of noise on 2% of samples, moves them by
    0.019–0.073, with worst row cosines down to 0.28.** This ViT amplifies any
    invisible perturbation that much (qimage's Q8.3 saw the same). Every stack
    decodes JPEG differently (llama.cpp uses stb_image), so this is noise, not a
    bug. Gate: reruns bit-identical (they are), and the decode gap within 2× the
    noise control.

- [x] **V4 — the tower on the device.** The tower's attention now runs on the
  matrix cores: the DiT's `dit_attention_wmma.comp` with `HEAD_DIM=80`, plus a
  `SRC_HEAD_DIM` define in `dit_pack_f16.comp` that zero-extends the 72-wide
  heads (every existing build's SPIR-V is byte-identical). It uses `OUT_F16`, so
  the context lands as the output projection's fp16 A operand, 80 columns a head.
  `padHeads` stages that projection's weight with zero columns at each head's
  pads, K = 1280. The padding is exact: zero components add nothing to q·k, and
  v's pad columns give zero context. wave32 when the device can pin the subgroup
  size, wave64 otherwise. New SPIR-V: `qvit_pack_hd80`, `qvit_attn_wmma_hd80{,_w32}`
  (`go generate ./shaders`). `VISION_ATTN=scalar` at construction keeps the old
  kernel as the control. **This applies to Qwen-Image's edit tower too.**
  - **Speed:**

    | image | patches | scalar | WMMA |
    |---|---|---|---|
    | 512² | 1 024 | 332 ms | **87 ms** |
    | 1024² | 4 096 | 6.0 s | **450 ms** |
    | 2048² | 16 384 | VK_TIMEOUT | **3.1 s** |
    | 4096² (the cap) | 65 536 | — | **32.8 s** |

  - **Gate `TestLLMGPUTower`**: five cards through one tower staged at 16 384
    patches, from 14×20 to 128×128 patches. The two large ones (`big` 1024²,
    `huge` 2048²) come from HF's SDPA, merged rows only.
  - **The bound, and why it changed.** The worst-element metric grows with size
    (rel 0.03 → 0.11 → 0.31), and the scalar kernel shows the same growth
    (0.12 at 1024²). So it's the fp16 tower, not the new kernel. A worst element
    also grows with the element count on its own. **The yardstick is the
    checkpoint's own bf16**, which the dump now runs against fp32 on every card:
    RMS-relative 0.027–0.10, worst row cosine 0.75–0.998. The device is
    **0.0015–0.0068 RMS and cosine ≥ 0.9964, 10–17× closer to fp32 than bf16**.
    Gate: RMS ≤ bf16/4 and worst cosine ≥ bf16's on every card, plus the old
    worst-element 0.1 on cards ≤ 1 024 patches.
  - **Arenas:** at the 65 536-patch cap the fp32 arena is 4.06 GB, inside the 4 GiB − 4
    binding limit with about 5% to spare. `allocActivations` now refuses a budget
    that would cross it rather than run fast and wrong
    ([[a-benchmark-that-gets-faster]]).
  - **Qwen-Image re-gated** (its vertical is parked, so this is a note, not a
    reopening). `TestGPUTower` passes: merged 0.135 against a bound of 0.25, and
    the wide card 0.21, where the device measured 0.562 before.
    `TestServedEditOracle256` passes: edited image **max abs 0.0033, mean
    0.00052** against a bound of 0.35, where 0.0014 / 1.7e-4 was recorded before.
    **A 1024² edit reference now encodes in 0.76 s, not 6.3 s.**
    `research/qimage-vertical.md`'s edit timings predate this.
  - Q2 (default max-pixels): **proposal**, a server default of **4 096 tokens an
    image (2048², about 3.1 s of tower plus about 3.4 s of prefill at ~1 200 tok/s)**,
    with a flag up to HF's 16 384 (32.8 s + ~14 s). It lives in V8/V9's flags.

- [x] **V5 — rotary positions decoupled from cells.** HF indexes its cos/sin
  rows **by cell**: the indexer ropes a pooled block with
  `full_cos.index_select(0, group_starts)`, and `group_starts` are cells. Every
  kernel here already reads the rotary table by cell (the pack at
  `SEQ_PAST + t`, a pooled block at its first cell). So **no shader changed.**
  Row c of the table now holds cell c's (t, h, w) angles. The table's own comment had
  predicted it: "the day an image batch makes them differ this table follows
  without an edit".
  - `llm/position.go`: `Pos3`, `ImageSpan` {Start, End, GridH, GridW, Hash},
    `cellPositions` / `cellDelta` / `truncateSpans`. `RoPEMulti3` takes a
    (t, h, w) per row, and `RoPEMulti` wraps it with identical arithmetic.
    `AttnLayerAt` is the CPU layer at given positions.
  - `AttnGPU`: **a rotary table a slot** (`RopeOff = wRope + slot·table`, 67 MB a
    slot at 262 144 cells), plus `SetRopeRows(slot, from, pos)`. Each table starts
    as the text table.
  - `Graph`: `Input` {IDs, Images []InputImage{At, GridH, GridW, Hash}}, with
    `AppendInputN`, `ExtendInput` and `HiddenExtendInput`. The old entry points wrap
    them. Image rows must hold the pad id (PLE's rule). Spans are per-sequence
    state carried beside `ids` through `UseSlot`, `Checkpoint`/`Restore` (which now
    also compare spans, hashes included, since the ids at an image are all the
    pad), `Reset`, `Rewind` and `rewindPosition` (a cut inside an image is
    refused). `prepRope` writes a pass's rows before it runs, in all three append
    paths (`appendIn`, the recorded step, and each batched row). It writes **only
    when the sequence has an image, or when the slot's table is dirty below
    `ropeHW` from an earlier one**. A text-only sequence in a clean slot writes
    nothing.
  - **Gates:**
    - `TestImagePositions`: 272 cells × 3 axes identical to HF's `get_rope_index`,
      delta −184.
    - `TestAttnGPUImagePositions`: 64 cells with a 6×8 image, device vs CPU at
      image positions. q, k, context and output rms 3e-5 to 4e-4. The indexer is
      exactly its text-position baseline (bf16 projections). Control: the text
      table misses by ≥ 100×.
    - `TestGraphImageIsAChunkSplit` (4 layers, a 12×16 image in 512 tokens):
      whole = 3 / 31 / 201 chunks **bit for bit**, the last ending in single
      tokens. Control: the same ids without the image move 2560/2560 values. A
      table dirtied by another image sequence is repaired for the next image
      and for text.
    - `TestGraphImageDecode`: 3 slots (two different image prompts, one text).
      `DecodeRows` is bit-identical to each solo step, and decoding after an
      image through the **recorded step** equals the prompt run whole.
    - `TestGraphRestoreKnowsItsImages`: same image restores; **a different image
      under identical ids is refused**; a cut inside an image is refused.
  - **Text unchanged**: `TestGraphLogits` reproduces its diagnostic digits
    exactly through all 48 layers (maxAbs 1.987 at 19208, rms 0.3314, argmax
    11751 vs 561, 0.4920 apart). **The whole `go test ./llm/` passes, 190 s.**
  - **A latent bug fixed on the way:** `PinSchedule` dropped only the live
    slot's recorded decode step, so a parked slot replayed a step recorded
    under the other schedule. Test-only in effect (the pin is an instrument),
    and found by `TestGraphImageDecode`.
  - **Q5 answered:** the MTP draft head keeps its own one-layer cache with a text
    table. `Speculate(true)` refuses a sequence holding an image, and a
    speculating graph refuses images.
  - **Constraint carried forward:** an image must fit in one pass (`Input`
    rows). The served `-llm-batch` is 8192, so V8/V9's per-image cap has to
    respect it. HF's 16 384-token maximum would need an image split across
    passes, which is not built.

- [x] **V6 — image rows into the graph.** `InputImage.Embd` ([gh·gw][2560],
  the tower's merged rows) replaces the pad's token embedding in the host-side
  upload. That is the whole scatter, because the hyper-connection init reads
  nothing else. The ids stay 248056, and `inputSpans` checks the row count
  before anything moves.
  - **The oracle is `reference/eval_dump_mtmd.c`**, eval_dump with an image
    through llama.cpp's mtmd. It writes the ids (pads at the image), `image.txt`
    (at, nx, ny), **mtmd's own embedding rows**, and the LLM's tensors per
    decode (mtmd decodes text / image / text as three calls; `traceRows`
    concatenates them). The Go side feeds mtmd's rows, so the comparison is the
    language model's alone. The fixture is the barn photo (ImageMagick's sample,
    500×500 → 16×16 = 256 tokens at cell 4, 281 cells). llama.cpp ends at
    n_past 41 = 281 − 256 + 16, HF's advance exactly. Oracles:
    `reference/out/llmvision_eval{,_depth,_moe}` and
    `llmvision_text281{,_depth}`, a 281-token text prompt as the drift
    baseline. The commands are in the file header; each run stages the whole
    model through llama.cpp and needs the machine.
  - **`TestGraphImagePrefix` (4 layers): `l_last-3` rms 6.3e-4** against
    `TestGraphPrefix`'s 5e-3. Controls: the pad's embedding instead of the rows
    is 656× the gate. **vLLM's text positions for the image 3.5×**, and a
    transposed grid 2.1×. Positions reach only one attention layer of four, so
    those margins are small here by construction. It still measures V0's ruling:
    llama.cpp's 3-D positions are ours, and vLLM's 1-D are not.
  - **`TestGraphImageLogits` (48 layers): the same argmax (1919, 18.2386 vs
    18.2452), 9 of the top 10 in common, logits 2.47% of scale** (bound 5%),
    result_norm 1.19% (bound 2%). The vLLM control lands 1.35× further off.
  - **Why an image prompt drifts ~4× as much as text of its length** (0.34% /
    0.57% for the 281-token text baseline), in full:
    - `TestGraphImageDepth`: the image prompt's *text* rows drift exactly as
      text does (0.353% vs 0.305% at layer 47), so nothing leaks. The *image*
      rows drift 3–6× more from **layer 0**, before any attention layer, so
      it's not positions or masking.
    - Per row: the typical image row is *nearer* llama.cpp than a text row
      (median 3.7e-3 vs 6.8e-3), but a handful jump to ~5% after layer 0.
    - `TestImageRoutingTies`: those are **MoE routing flips**. 20 of 256
      image rows (0 of 25 text) take one different expert, every one at a
      near-tie in the reference's own router (10th-to-11th gap 1e-4 to 2.6e-2).
      **Our router is more accurate on image rows than on text** (rms-rel 1.0e-3
      vs 2.6e-3). Image tokens simply route on tighter margins: median gap
      0.019 vs 0.054, 10th percentile 0.002 vs 0.010. Any two implementations
      flip some, bf16 HF included. `LLM_DENSE_FP16=1` changes nothing (1.146%),
      so it isn't the bank either.
  - **End to end, `TestGraphImageOwnTower`: our decoder, processor and device
    tower** give rows within **rms-relative 0.025 of mtmd's** (inside V3's
    invisible-noise range), and through the whole model **the same argmax
    and 8 of the top 10**. Greedy from there, through our whole pipeline:
    *"This picture shows a rustic wooden barn in a green meadow, with the
    majestic, snow-capped Teton Range rising dramatically in the background
    under a clear blue sky."* (It is the Moulton Barn.)

- [x] **V7 — the template and the token stream.**
  - `ChatMessage.Parts` ([]ChatPart{Text, Image}, the template's content
    list) and `ChatOpts.AddVisionID`. An image renders as `ImageMarkup`
    (`<|vision_start|><|image_pad|><|vision_end|>`) in place, "Picture N: "
    before it under `add_vision_id`, numbered across the conversation by the
    main loop only (the template's last-query scan does not count, and
    renders N as 0, as the template does). A system turn with an image is
    refused.
  - `ExpandImages(ids, pad, []PromptImage)` widens the i-th pad to image i's
    gh·gw pads after tokenizing, which is the processor's order reversed and
    the same stream (the pad is a special token). It refuses a pad count that
    is not the image count, so text that spells `<|image_pad|>` cannot
    conjure a picture.
  - **Gates:** `TestRenderChat` now has 28 Jinja cases, five with images: an
    image before text, images between text with trimming, `add_vision_id`
    across turns, an image inside a tool response (the last-query scan), and
    thinking off. All are character-identical.
    **`TestExpandImagesIsTheProcessor`**: the dump's two-image message renders
    to HF's `apply_chat_template` output exactly (424 chars), and tokenized and
    widened it is **the processor's `input_ids`, 272 ids, id for id**, both
    images at the processor's cells and grids.

- [x] **V8 — the API doors.** Each door lands an image as the one form the
  backend reads, an `image_url` block holding a `data:` URL, in its place
  between texts:
  - chat `image_url`;
  - Responses `input_image` (`image_url` string; `file_id` refused, no files);
  - Messages `image` (base64 source becomes a data URL; a url source passes
    through and is refused).
  **Q3 decided (default): remote URLs are refused** with the fix in the message
  (send the bytes). Audio, video, files and documents stay refused with
  specific reasons. So do images in a tool result or a system prompt (the
  template refuses the latter), and WebP (Q6). `MessageContent.Unreadable`
  replaced `NonText`. **Gates:** `TestImagesReachTheBackend` (all three doors
  deliver text / image / text in order, as a data URL) and the updated refusal
  tables. **`API.md` documents it** (the "Images in a conversation" section,
  the flags, and the table).
- [x] **V9 — serving** (the open items are below). `backend/llm_vision.go`:
  - `-llm-mmproj` stages the tower beside the LLM (+0.86 GB weights and
    ~1.3 GB activations at the 4096-token cap). `-llm-vision-tokens` (4096,
    held ≤ `-llm-batch`, so an image is one pass; this is Q2's proposal as the
    default) and `-llm-vision-images` (8). Off by default: turning it on in
    `ai.service` is a deployment decision.
  - A request's images are decoded and encoded **before it takes a slot**,
    under the device lock. `ExpandImages` then builds the `llm.Input`.
  - **The scheduler matches keys, not ids.** An image cell's key is negative
    and derived from the picture's hash and the cell index. `held` and a new
    `llmSlot.ckptKeys` are keys, so a picture's reuse is by picture. A prefill
    chunk is never cut inside an image. Image chunks go through `ExtendInput`,
    and text chunks keep the old calls.
  - **The checkpoint mark** is found on the single-pad prompt and moved past
    each image (`expandedMark`), and its size threshold is the expanded one (a
    bug caught by the gate: 256 tokens of prefix held as one pad looked too
    small to checkpoint).
  - **An encoded-image cache** (256 MB, LRU, by the bytes' hash). A follow-up
    turn resends the picture and pays no tower.
  - The request log line gains "(N images, T of tower)". `queued` now counts
    only the wait for a slot.
  - **Gates:** `TestPromptKeys`; `TestVisionCacheEvictsOldest`; and
    **`TestLLMImageCheckpoint`** (4 layers): a turn after an image restores
    **323 tokens from a checkpoint past a 256-token image**, twice, streaming the
    fresh prefill's text exactly. **The same picture mirrored, which gives
    identical ids, is not restored.** `TestLLMCheckpointIsTheSystemPrompt` is
    unchanged.
  - **Through the real server** (48 layers, 2 slots, `-llm-ctx 8192`):
    - barn 256 tokens, tower 53 ms, "a barn in front of a mountain";
    - the follow-up **reused 286 cached tokens, image included**, and named
      the Teton Range, Wyoming and Mormon Row;
    - a 1920×1080 login-screen image, 2 040 tokens and 0.77 s of tower,
      prefill 1 045 tok/s, read *"Friday, 15 February 2013 14:52 PM"* off it;
    - the Messages door ("wooden barn") and the Responses door ("neither",
      for "nature or city?" of a login screen);
    - streaming an image;
    - a 2592×1754 photo at the 4096 cap, 2.3 s tower + 3.4 s prefill, then a
      follow-up with **0 s of tower, 4 058 of 4 079 cached, ttft 0.25 s**;
    - WebP and remote URLs refused.
  - **Open, measured-first:**
    - ~~**The tower holds the device lock for a whole image.**~~ V12: it
      yields every 50 ms now.
    - **An image larger than a pass** (HF allows 16 384 tokens) would need an
      image split across passes. Not built: the cap keeps it out.
    - **`add_vision_id`** is still refused as a template kwarg by the API's
      allow-list, although `RenderChat` supports it. It's a one-line
      pass-through.

- [x] **V10 — does it see. Yes, and exactly as llama.cpp does.**
  `reference/eval_llm_vision.py` (`gen` / `run` / `score`) builds 18 fixed items
  from a seed and asks them of any OpenAI-compatible server. Requests are greedy
  (`temperature 0, top_k 1`), `enable_thinking: false`, and sent through one
  client. The oracle is **llama-server on the same GGUF and mmproj**
  (`--jinja --image-max-tokens 4096`, command in the script header), which
  compares the two stacks door to door. Items: two photo descriptions, two
  readings off the login screenshot, OCR (a prose paragraph, a dense 17 px
  invoice, and six random codes a prior cannot guess), a bar chart read
  value by value, tallest/shortest on an unlabelled chart, counting red
  circles among blue distractors (4, 7, 12), shape order on a 1400×360 and a
  360×1200 canvas, a 3×5 letter grid (one cell, then every row), and two-picture
  requests (which shape changed, and circle counts in each). The
  non-square canvases and the grid are the transposed-grid and
  position-bug catchers. Results land in `reference/out/llmvision_v10/`
  (ignored), so the table below is the record.

  | | ours | llama.cpp |
  |---|---|---|
  | items passed | **17/18** | **17/18** |
  | OCR exact (paragraph, codes, grid rows) | 3/3, CER 0 | 3/3, CER 0 |
  | the one miss | `ocr_small`: lines 2–3 merged, CER 0.003 | the same miss, same text |
  | prompt tokens, all 18 requests | identical to llama.cpp's | |
  | text identical to the other | 15/18 | |
  | wall for the set (1 slot) | **34.9 s** | 90.8 s |

  The three answers that differ are free-form sentences ("A tilted photo
  shows…" vs "A rustic wooden barn sits…"), and both pass. The miss is a
  formatting choice both stacks make identically, not a misread: every character
  is right. **The set saturates.** Both stacks pass everything they can, so it is
  a regression gate for the image path (rerun it after any change to the tower,
  positions or scheduler), not a quality benchmark. Prompt tokens are equal
  everywhere, from 143 (the codes card) to 4 025 (the 2592×1754 photo at the cap),
  so our smart-resize and llama.cpp's agree on every grid here.
  Speed on the same requests: the tower is 28 ms (the codes) to 2.1 s (the
  photo at the cap), and prefill of the image prompts runs 300–1 290 tok/s.
  An image already in the encoded-image cache costs 0 s of tower (`ocr_screen_user`,
  `grid_rows`).

- [ ] **V11 — video (parked until V10).** `<|video_pad|>`, the temporal
  patching of real frame pairs, timestamps between frames (Qwen3.5-style
  `<t> <vision_start> frame <vision_end>`), frame sampling, and the video
  processor's own size limits.

- [x] **V12 — the tower shares the device.** V9 left it open: the tower ran
  as one `Device.Do`, so an image held the device for all of its tower.
  - **The harness.** `cmd/loadgen` gained image streams (`-image`,
    `-image-at`, `-image-max`, `-image-class`), a **max gap** column (the
    longest wait between two deltas), and a `stall:` line for every gap over
    `-stall` (0.3 s), with its time. Each image request inserts a counter in
    a JPEG comment segment, so no two share bytes and the encoded-image cache
    never skips the tower. *Changing one pixel does not do this*: q95
    quantization erased it, and 3 of 4 requests in the first try hit the cache
    ("0s of tower").
  - **The arm.** 3 slots, `-llm-ctx 65536 -llm-batch 8192`. One background
    agent (7 338-token prompt, 768 out) at t=0. Two interactive image requests
    at the cap (the 2592×1754 photo, 4 024 tokens, ~2.1 s of tower) at 12 s and
    20 s, which is during the agent's decode. A voice command at 12.5 s, landing
    inside the first image's tower. Two runs an arm, which agree within 0.03 s.
  - **Two stalls, measured apart:**
    1. **The tower**, 2.1 s under the lock. The voice command's TTFT went from 0.22 s
       solo (restored from its checkpoint) to 1.97 s.
    2. **The image's prefill**, 4 024 rows in one uncuttable pass (V5: an image
       fits one pass), 3.15 s. It stops the voice command *mid-answer*
       (a 3.15 s gap, total 0.68 → 5.7 s).
  - **The change, in three parts:**
    - `vision.GPU.Between` is called between submits (every 8 dispatches, and
      nothing of the tower's is on the device then).
    - The backend's hook calls `Device.yield` once `LLM_TOWER_SLICE` (**50 ms**)
      has passed. `yield` hands the device to the waiters queued now, one `Do`
      each, and takes it back ahead of their next. That needs **a FIFO device
      lock** (`fifoLock`). On a `sync.Mutex` the unlocker barges straight back,
      or a waiter finishing one `Do` barges ahead for its next.
      `TestDeviceYieldHandsOff` hung on the mutex, and passes 20× under `-race`
      on the FIFO lock.
    - **An interactive request's tower counts as an interactive job for rule 1**
      (`llmSched.towerBegin`), so a background prefill chunk (up to ~2 s)
      never lands in a gap and stretches the tower by that much. Gate:
      `TestTowerHoldsBackground`. A new mutex, `llmVision.tower`, keeps a
      second image out of the arenas while the device is lent.
  - **Sweep** (`LLM_TOWER_SLICE`, one server an arm, both runs shown as one;
    they agree):

    | | 0 (control) | 100 ms | **50 ms** |
    |---|---|---|---|
    | voice TTFT, landing in a tower | 1.97 s | 0.28 s | **0.28 s** |
    | voice total (solo 2.4 s cold, 0.68 s restored) | 5.74 s | 5.60 s | **2.00 s** |
    | voice's worst gap | 3.15 s | 3.15–3.22 s | **0.19 s** |
    | that tower's wall (alone 2.09 s) | 2.09 s | 2.60 s | 2.79 s |
    | the image's TTFT | 5.59 s | 5.92 s | 6.10 s |
    | background agent's stall in a tower | 2.27 s | 2.27 s | 2.25 s |
    | aggregate tok/s | 21.49 | 21.37 | 21.35 |

    At 50 ms the voice command decodes at ~10 tok/s *through* the tower and
    finishes before the image's prefill starts. At 100 ms it was still
    decoding when the prefill began, and ate stall 2. The image pays for its
    neighbour: +0.7 s of tower wall, which is roughly the voice command's
    own device time. The background agent is held out, as rule 1 says, so for
    it nothing changes.
  - **Gates:** `go test ./backend ./qimage/vision` pass, and the V10 eval
    through the new build is 17/18 and **18/18 texts identical** to V10's run.
  - **Still open: stall 2**, an image's prefill as one pass. It is 3.15 s at
    the cap, and an interactive text prompt of 4k tokens stalls a voice command
    the same way (only background prefill is chunked, rule 3). Only images
    can't be cut, though, and that is V5's "an image split across passes",
    not built. Worth it only if interactive image requests and voice commands
    really coincide on the served machine.

## Open questions

- **Q1** Does `qimage/vision` move to a shared `vision/` package, or does `llm`
  import from `qimage`? (Decide in V1.)
- **Q2** Server default max-pixels. HF allows 16 384 tokens an image.
  **Defaulted to 4096** (`-llm-vision-tokens`, a 2048² picture, 3.1 s of tower)
  from V4's curve; yours to change.
- **Q3** Remote image URLs: fetch or refuse? **Refused by default** (V8); a
  fetch would be a flag with size and time caps.
- **Q4** Is the torchvision bicubic path actually what the *fast* processor
  runs on a uint8 PIL input, or does it go through Pillow first? Settled by
  reading `image_processing_qwen2_vl_fast.py` in V3.
- **Q6** WebP. The standard library cannot decode it, and this module has
  no dependencies. Take `golang.org/x/image/webp` (the first dependency), or
  keep refusing? Clients mostly send PNG/JPEG. Refused today.
- ~~**Q5** How does the MTP draft head take positions?~~ It doesn't; it is
  refused behind an image (V5).

## Handoff

**State at 2026-09-24, session 2:** V0–V10 and V12 are done. The server answers
images through all three doors, gated end to end. V10's eval scores 17/18,
identical to llama-server's on the same weights, and the one miss is shared.
Nothing is committed (the auto-committer owns main).

**What changed outside new files, for review:**
- `llm/graph.go`: the `Input` API, spans state, `prepRope`, `PinSchedule` now
  drops every slot's recorded step, and `Speculate` refuses an image sequence.
- `llm/gpu_attn.go`: a rotary table a slot. `llm/attn.go`: `RoPEMulti3`,
  `AttnLayerAt`. `llm/chat.go`: `Parts`, `AddVisionID`.
- `qimage/vision/gpu.go`: matrix-core attention. **This changes Qwen-Image's
  edit path too, re-gated** (V4).
- `shaders/dit_pack_f16.comp`: a `SRC_HEAD_DIM` define. Existing SPIR-V is
  byte-identical; the new `.spv` needs `go generate ./shaders` (gitignored).
- `backend/llm.go`, `backend/llm_sched.go`: images, keys, checkpoint mark,
  cache. `cmd/serve`: three flags.
- V12: **`backend/device.go`'s lock is now a FIFO lock** (every vertical's
  `Do` goes through it) plus `yield`. `backend/llm_sched.go`: `towers`
  in rule 1. `backend/llm.go`: the class and scheduler are read before the
  images. `qimage/vision/gpu.go`: `GPU.Between`, nil for Qwen-Image.
  `cmd/loadgen`: image streams and stall columns.
- `api/{chat,completion,messages,responses}.go`, `API.md`. **Another session's
  uncommitted edits were already in `API.md`, `api/completion.go`,
  `api/messages.go` and `api/presets*.go`** (preset/thinking wording, from
  01:10 on 2026-09-24). They were left as they were, and mine are in other
  functions.

**Reproduce:**

    reference/fetch_llm_checkpoint.sh
    .venv/bin/python reference/dump_llm_vision.py          # HF oracle, ~1 min
    # llama.cpp oracles: reference/eval_dump_mtmd.c header (whole model each)
    go test ./qimage/vision ./llm/pixels ./api ./backend
    LLM_BANK_CACHE=$PWD/models/Qwen3.8-Flash-Next-GGUF/bank-cache go test ./llm/ -timeout 60m
    go run ./cmd/serve -llm -token= -llm-ctx 16384 -llm-mmproj models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf
    .venv/bin/python reference/eval_llm_vision.py gen
    .venv/bin/python reference/eval_llm_vision.py run --name ours     # ~35 s, whole model
    # stop it, start the llama-server line in the script header, then:
    .venv/bin/python reference/eval_llm_vision.py run --url http://127.0.0.1:8081 --name llamacpp
    .venv/bin/python reference/eval_llm_vision.py score ours llamacpp

The V12 arm (server at `-llm-ctx 65536 -llm-batch 8192 -llm-slots 3` with
`-llm-mmproj`, arms by `LLM_TOWER_SLICE`):

    go run ./cmd/loadgen -agents 1 -voice-at 12.5s -image-at 12s,20s -runs 2

**Next, in order:** `add_vision_id` pass-through; Q1/Q6 when the user
decides; then V11 (video), which V10 unparks. Stall 2 (V12) only if it shows up
in use.
