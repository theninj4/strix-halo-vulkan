# G0 — instruments and the ceiling (2026-09-29)

*The G0 stage, broken out of `KERNELS.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`kernels-vertical.md`](kernels-vertical.md).*


Filed as **§0.6** in `research/0-measurement-validity.md`.

**The ceiling is the hardware's.** The `peak` family gained the WMMA loop
at 8 and 16 chains and at wave32 (`-DACC`, `-DWAVE`; `bench/ops_peak.go`
pins the size), so G-o1 could be answered instead of assumed:

| probe | best | sclk | W | FLOP/clk/CU |
|---|---|---|---|---|
| wmma_fp16, 4 chains, wave64 (the old probe) | 55,554 | 2890 | 112 | 481 |
| 8 chains | 55,615 | 2890 | 99 | 481 |
| 16 chains | 55,524 | 2890 | 98 | 480 |
| wave32, 4 chains | 55,436 | 2892 | 98 | 479 |
| wave32, 8 chains | 55,405 | 2896 | 100 | 478 |
| wmma_int8 | 55,538 | 2879 | 98 | 482 |

Flat to 1% across all of them, at **480 of the spec's 512**, which is
15/16 exactly: the matrix pipe's dense rate on this silicon, not a probe
artefact. 55.5 TFLOP/s at 2899 MHz stands, and every rate below is read
against 480 FLOP/clk/CU at the clock the kernel ran at.

**The clock under a real kernel.** A 100 ms sampler on
`hwmon/freq1_input` and `power1_average` around `TestGPUShapes`
(`H3_SHAPES=1 H3_PROFILE=1`, int8, the machine otherwise idle):

| what | sclk (mode) | package W | a 480p forward | a 768p forward |
|---|---|---|---|---|
| `peak` probe | 2890–2899 | 100–112 | | |
| suite GEMM, 20 ms bursts (`results/gemm_wmma.csv`) | 2813 | 137 | | |
| **H3 forward, CPU idle** | **2600–2650** | **149** | 30.0 s | |
| H3 768p forward, CPU idle (sustained ~2 min) | 2550–2600 | 142–155 | | 106.7 s |
| **H3 forward, 32 busy CPU threads** | **1300–1400** | 121 (GPU side) | **50.1 s** | **190 s** |

Three things. **The package is power-limited at ~150 W**, and a GEMM's
memory traffic costs ~40 W over the register-only probe, so the clock
sags 9–10% under any real kernel: the ceiling during a video forward is
**49.9 TFLOP/s**, not 55.5, and the shipped GEMM's 37 TFLOP/s there is
**74% per clock** (353 FLOP/clk/CU), the attention's 37.3 is 75%, the
whole forward's 32.5 is 65%. **The 768p run sags further** (2550–2600),
so long forwards are ~2% slower per FLOP than short ones for the clock
alone. And **the CPU shares the budget**: 32 busy host threads halve the
GPU clock and make a forward 1.67x slower. That is the mechanism behind
VIDEO.md M11f's "+2.4 s of device time while the audio decodes on 32
threads": not memory contention, the power budget. Anything the server
runs on the CPU beside a device job costs GPU clock in proportion to its
power, which is a serving rule this repo did not have.

**Instruments.** `cmd/probe` is back (`RADV_DEBUG=shaderstats|asm`),
`H3_GEMM_SPV=path` runs a big-GEMM build from disk in `TestGPUShapes`
(and times it even when it is wrong by construction), and the sampler
is `scratchpad/sample.sh`-shaped: 100 ms of `freq1_input`,
`power1_average`; a phase is the samples between two timestamps. It is
not in the tree yet; G7 should fold it into `bench/sysmon.go` around the
vertical tests.
