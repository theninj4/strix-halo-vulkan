# S8 — the transducer tail on Vulkan

*The encoder projector, the prediction network, the joint and the TDT greedy
loop of `parakeet-tdt-0.6b-v3`, ported from the CPU reference to the device.
SPEECH.md's eighth stage; the code is `parakeet/gpudecode.go` and the four
`shaders/parakeet_lstm_*` / `parakeet_joint_sum` / `parakeet_argmax` files, and
it is validated by `parakeet/gpudecode_test.go`.*

**Headline: the decode loop goes from 208 ms to 4.8 ms — 43x — and an 11 s
clip goes from 255 ms to 43 ms, 257x real time. The transcript, the 46
emissions, their frames and their durations are unchanged. What made it
possible is that the whole tail is 24 MB and the last-level cache is 32.**

| | host (S7) | device (S8) |
|---|---|---|
| projector + TDT loop | 208 ms | **4.8 ms** |
| front end (host FFTs) | 21 ms | 21 ms |
| encoder | 20 ms | **17 ms** |
| whole pipeline | 255 ms | **43 ms** |
| real time | 43x | **257x** |
| transcript | exact | **exact, identical trace** |

## The problem is not arithmetic, and it is not a tile

Every other kernel in this repository is a throughput problem. This one is
not. Greedy transducer decoding is **sequential by construction**: the
prediction network is a language model over the tokens already emitted, so
step n+1's input is step n's output and there is no batch anywhere. An 11 s
clip is 46 emissions of 24 MFLOP each — 1.1 GFLOP, which the encoder does in
80 µs — and it took 208 ms, because every one of those emissions was a round
trip between a host holding the decision and a device holding the model.

So the design is entirely about what crosses the bus and how often:

- **the model stays resident.** 24 MB of fp16: both LSTM layers, the
  prediction projector, the 8198-wide joint head and the encoder projector.
- **the state stays resident.** h and c for both layers, the projected
  encoder frames, the logits.
- **the hidden states never come back.** `GPUDecoder.Attach` builds one extra
  pipeline — the same narrowing shader over a descriptor set whose activation
  binding is the *encoder's* buffer — so the `[138, 1024]` tensor between the
  two stages stays on the device. Measured, not assumed: that read is 552 KB
  and takes **2.4 ms**, because device-local host-visible memory reads at 0.2
  GB/s on this part (stage 3c's finding, which was the same 0.2 at 63 MB). It
  would have been half again what the entire loop now costs.
- **what does cross** is 640 floats of embedding in (2.5 KB, a host-side
  gather into the arena — putting an `[8193, 640]` table on the device would
  cost 21 MB to save 0.2 µs) and **two floats out**: the argmax over the
  vocabulary and the argmax over the durations.

## No new GEMM

The four projections run on `dit_gemm.comp`, the same rungs the encoder does,
with M padded from 1 up to the narrowest tile in the ladder. That looks
wasteful and is not, and the reason is worth stating: **a GEMM reads its B
operand once whatever M is.** At M = 1 the weight is the entire cost, and
padding M to 16 buys 16x the arithmetic on a part that has arithmetic to
spare.

`TestGPUDecLadder` confirms it directly, and it is the cleanest evidence in
the file:

    kernel                  one emission
    reg32x32_bt16_w32          73 µs   1.00x
    reg16x64_bt16_w32          73 µs   1.00x
    reg16x64_bt16              78 µs   1.08x
    reg32x64_bt16_w32          86 µs   1.18x
    reg32x64_bt16             106 µs   1.45x
    reg32x128_bt16            116 µs   1.59x
    reg64_bt16                147 µs   2.02x

**BM = 32 ties BM = 16 exactly, twice, reproducibly** — doing double the
arithmetic for the same time. That is the statement that the tail is a
bandwidth problem at M = 1, and it is why a dedicated GEMV kernel is not worth
writing here: it would win back arithmetic nobody is paying for. (The wave64
rungs still lose, at 1.08-2.02x. That is §6.2 for a fourth set of shapes.)

`DefaultDecPlan` takes the narrower of the two tied rungs, on the principle
that the one doing less work is the safer default on a part that might not
have the arithmetic spare.

## The finding: 24 MB against a 32 MB cache

One emission is **65 µs on the device**, against 105 µs of wall clock:

    lstm in           1.5 µs
    gemm lstm0       14.6 µs     6.6 MB of weights   449 GB/s
    lstm gate         0.8 µs
    lstm in           1.5 µs
    gemm lstm1       14.6 µs     6.6 MB              449 GB/s
    lstm gate         0.8 µs
    narrow lstm       1.1 µs
    gemm pred.proj    7.6 µs     0.8 MB              108 GB/s
    joint sum         1.5 µs
    gemm joint       15.4 µs    10.8 MB              702 GB/s
    argmax            7.9 µs

Read the bandwidth column. 449 and 702 GB/s are both **above this part's DRAM
bus** and below §5.1b's 930-965 GB/s MALL ceiling: the tail is being re-read
out of the last-level cache, 46 times, because 24 MB of weights fits inside
the 32 MiB §0.4 measured. At DRAM speed the loop's weight streaming alone
would be 828 MB at ~250 GB/s = 3.3 ms of pure bandwidth; it is 3.0 ms of
*everything*.

This is the first place in the engine where §5.1b's 32 MiB is a **design
constraint that was met** rather than a cliff that was fallen off. It also
says what the tail's scaling limit is: a bigger prediction network, or a
second hypothesis in a beam, and the whole loop drops to the DRAM law.

The two small kernels are the other half of the story. `gemm pred.proj` moves
one thirteenth of the joint's bytes and takes half its time, because a
`[640, 640]` weight at BN = 64 is **ten workgroups** on a 40-CU part; `argmax`
is one workgroup reading 33 KB. Together they are 24% of an emission and
almost none of it is memory. Both are fixed costs of a dispatch that cannot
fill the machine, which is what "latency bound" means concretely.

## What is still on the host, and what it costs

40 µs of the 105 µs per emission — **38%** — is the submit and the fence. The
loop is one command buffer per emission because the decision that chooses the
next one is made on the host: the two argmaxes come back, the cursor advances
by the predicted duration, and a blank asking for a zero-frame jump is forced
to one so the loop cannot stall.

That is the honest first version and it is deliberately not the last. Two
things would remove it:

- **a persistent kernel**, with the loop's control flow on the device and a
  grid-wide barrier between stages. It is what a streaming API would want —
  `api/transcription.go`'s `transcript.text.delta` is already defined — and it
  is a different kind of kernel from anything here.
- **speculating on blanks.** A transducer emits runs of blanks, and during one
  the prediction state does not change, so the joints for frames t, t+1, ...
  are independent and could be one GEMM at M = 16 for the price of the M = 1
  one. This clip only has 8 blanks in 46 emissions, so it would buy little
  here; on a clip with silence in it, it would buy a lot.

Neither is worth doing until something else is the pipeline, and something
else already is: **the front end is 21 ms of the 43**, a few thousand
512-point FFTs in float64 on the host.

## Two folds, and what they cost a test

The prediction projector's bias is folded into `parakeet_joint_sum.comp` and
the joint head's bias into `parakeet_argmax.comp`, because both of those
passes already read every element of the vector the bias is added to. On a
loop that runs once per emitted token, each fold is a dispatch saved per
token.

It is worth recording that this was the only thing in the stage that looked
like a bug and was not. The arena holds `W·x`, the reference holds `W·x + b`,
and for the joint head that is a **4.8% relative** difference — most of that
bias is around -6.3 — which is exactly the size of a real error. The test adds
the bias back, with a comment saying why.

## Validating it

The prediction network is walked through two steps, because one would not
exercise the recurrence at all *and* would hide a swapped gate order: the
blank's embedding row is zero in this checkpoint, so the first step's gates
come entirely from the bias.

    dec_out    1.8e-04 relative     dec_out1   3.5e-04
    dec_h      1.0e-05              dec_h1     6.4e-04
    dec_c      5.1e-05              dec_c1     4.5e-04
    encoder_projected  1.9e-04
    joint_logits       4.0e-05

The last row is the one to read. An independent numpy computation of the same
product with fp16 operands and fp32 accumulation puts the floor at **3.3e-5**;
the device measures 4.0e-5. There is nothing in that number but the rounding.

The end-to-end assertion is the trace, for S6's reason: the decisions here are
discrete, so either every emission agrees or the port is wrong. It runs twice
— once over the reference's own encoder output (`TestGPUDecodeMatchesReference`,
which isolates the tail) and once over the device's (`TestGPUPipeline`, the
whole vertical) — and both produce 46 emissions at the same frames with the
same durations and the same string.

## What this leaves

    front end     21 ms   48%   <- host FFTs, float64
    encoder       17 ms   40%   (14.3 ms on the device)
    decode         5 ms   12%
    total         43 ms          257x real time

The model is finished. From 2.68 s on the CPU to 43 ms, 62x, with the
transcript unchanged at every step of the way — and the thing that is now half
the pipeline is the one part of it that has never been anything but a
straightforward CPU implementation.
