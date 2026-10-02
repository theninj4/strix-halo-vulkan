# The cache-size cost in prefill (done 2026-09-23)

*Broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


**The indexer's score wrote one float a token for every pooled block the
cache holds, not the context.** Every block at or past `nBid` (the first
that is not whole) pools cell 0 and so shares one score, and P7 already
computed that score once. But `llm_attn_score_wmma.comp` then *filled* it
into all `nBlocks - nBid` entries of the row. That is 65 535 floats a token
at depth zero in a 262 144-cell cache, **512 MB a layer for a 2048-row
pass**, streamed out at ~24 GB/s. Nothing on the device reads those entries:
P14-1's block select stops at the last block a cell can reach, and the
expansion (the cell-select control arm) maps every tail cell to `deadBid`.
Only the host's `AttnGPU.Score()` readback, which the tests compare with the
reference's whole `indexer_score_tokens`, ever saw them.

So the kernel writes the one entry at `nBid`, and `Score()` fills the rest on
the host, so the tensor keeps the reference's shape. The scalar control
kernel (`LLM_ATTN_SCORE=scalar`) got the same change, so the two arms keep
one contract. `-attn -tokens 2048` found it in one run: `score` was 4.46 ms
a layer at 65k cells and 20.6 at 262k. After the fix it is **0.29 and 0.30**,
because the fill was most of the kernel at *any* cache size.

**It is bit-exact.** The selection reads the same scores. `TestGraphLogits`
reproduces its diagnostics to the digit through 48 layers (maxAbs 1.987e+00
at 19208, rms 3.314e-01, argmax 11751 against 561). `TestAttnGPUIndexer`
and `TestAttnGPUIndexer4k` compare the full tensor, tail included, against
the reference, and they pass because of the host fill. The attention, QSA
and graph gates pass (`-run 'TestAttn|TestQSA|TestGraphIsAChunkSplit|
TestGraphSlots|TestGraphDecodeRows'`, 49 s).

**At 48 layers**, shipped banks, `cmd/llm -graph -tokens 2048`, the old and new
binaries interleaved, two passes each (agreeing to 6 ms):

| cache | before | after | attention before → after |
|---|---:|---:|---:|
| 262 144 cells | 1922 / 1916 ms (1067 tok/s) | **1696 / 1696 ms (1208 tok/s), +13%** | 365 → 149 ms |
| 65 536 cells | 1728 / 1723 ms (1187 tok/s) | **1688 / 1688 ms (1213 tok/s), +2.3%** | 178 → 143 ms |

The 262k cache now costs 8 ms more a pass than the 65k one, where it cost
194 ms.

**Against depth** at the served cache (`-depth -depths 0,64000,128000,192000
-pp 2048 -ctx 262144`, one sweep each). The fill was `nBlocks - nBid` a
token, so the gain shrinks as the context fills:

| depth | pp before | pp after | |
|---:|---:|---:|---:|
| 0 | 1043.4 | **1170.2** | +12.2% |
| 64 000 | 984.4 | **1050.8** | +6.7% |
| 128 000 | 953.2 | **968.1** | +1.6% |
| 192 000 | 892.2 | **901.7** | +1.1% |

**Decode did not move, and the prefill gain reproduces.** The sweep's
4-token decode read 0.2-0.6 tok/s lower in the new arm, so three
interleaved pairs of `-depths 0,64000 -tg 128` settled it:

| | pp at 0 | pp at 64 000 | tg at 0 | tg at 64 000 |
|---|---:|---:|---:|---:|
| before (3 runs) | 1035.4-1038.6 | 984.7-986.0 | 34.06-34.12 | 32.61-32.82 |
| after (3 runs) | **1170.6-1174.6** | **1046.4-1048.0** | 34.13-34.18 | 32.72-32.79 |

Decode is +0.07 / +0.05 tok/s, inside the spread. A decode step's tail fill
was 65k floats a layer, which is small but not nothing, and the dead score is
now computed by one workgroup rather than every stripe. The 237k agent
prefill C6 measured at 901 tok/s is not re-measured. Linear in the tail,
the saving averages roughly half of depth zero's 226 ms a chunk over a
0-237k walk, so an estimated ~5%.

**What still scales with the cache, and is left:** `idx` at `past == 0`
pools every block of the table (0.09 → 0.31 ms a layer, once a sequence),
because a later pass's dead block reads that fill. `select` is 0.21 →
0.38 ms a layer, probably the score rows' wider stride. Together they are
the 8 ms above.
