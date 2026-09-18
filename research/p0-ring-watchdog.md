<!-- LLM.md P0. The >2560-row stall: a ring watchdog, a force-signalled fence
     and an unbounded poll. Cited from vk/shim.c, llm/record.go and llm/graph.go. -->

[← LLM.md](../LLM.md) · [← LLM2.md](../LLM2.md) · [research index](README.md) · [L7d](l8d-moe-decode.md) · phase 2

# P0 — the stall was a two-second ring watchdog, and the hang was a poll with no timeout

**Result: the cliff is not rows, not layers, not bytes and not residency — it
is GPU time in one submit, and the limit is 2 seconds.** A command buffer that
holds the gfx ring longer than that is killed by amdgpu's watchdog; the reset
*force-signals the fence*, so the submit returns successfully and the only
evidence left is the timestamps the killed dispatches never wrote. Reading
those back with `VK_QUERY_RESULT_WAIT_BIT` is an unbounded **userspace** poll
inside RADV, and that poll — not the submit, not the driver's ioctl — is the
100%-of-a-core spin that looked like a hang.

Chunking a pass by time instead of by dispatch count fixes it. `-graph` now
returns at every row count tried, and **prefill does not plateau**: 4096 rows
is 1174.6 tok/s and 8192 rows is **1213.5 tok/s, 3.10x llama.cpp**, which is
the best prefill figure in this vertical.

## What the previous diagnosis had wrong

L8c-7 recorded the stalled goroutine as being in `Graph.flush → recorder.submit`,
in `[syscall]`, "which places it before the wait, inside the driver's submit",
and suspected residency: ~82 GB of pinned buffers, a 28.8 GB mmap'd n-gram
table, arenas that grow with the context, and a re-validation on every submit.

Three of those are wrong, and the first one is why:

1. **`[syscall]` is Go's label for a cgo call, not for a kernel syscall.** The
   thread's own accounting settles it: `utime=5645`, `stime=5` — 56.45 s of
   *user* time against 0.05 s of *system* time, `wchan` 0, state `R`. It was
   never in the kernel. Sampling `/proc/<tid>/stat` for the duration of a hang
   shows `stime` frozen and `utime` climbing exactly 100 jiffies a second.
2. **It is not residency.** Through the whole hang `mem_info_gtt_used` is flat
   at 80.198 GB, `mem_info_vram_used` flat at 7.423 GB and `VmRSS` flat at
   31.56 GB. Nothing allocates, nothing faults, nothing is evicted.
3. **It is not the size of the staging.** L8c-7 saw 2560 rows succeed in a
   staging sized for 4096. 3072 rows hangs just as hard in a staging sized for
   3072, so the arenas are not the variable either.

A native backtrace names it exactly. `ptrace_scope` is 1 here, so gdb has to
be the ancestor — launch the run under `gdb --args`, `handle SIGUSR1 stop`,
and signal the inferior once it hangs:

    #0  clock_gettime ()                             <- vDSO: no syscall
    #1  clock_gettime () from /usr/lib/libc.so.6
    #2  ?? () from /usr/lib/libvulkan_radeon.so
    #3  ?? () from /usr/lib/libvulkan_radeon.so
    #4  ?? () from /usr/lib/libvulkan_radeon.so
    #5  shim_dispatch_multi_timed (..., count=1024, ...) at shim.c:941
    #6  _cgo_..._Cfunc_shim_dispatch_multi_timed
    #7  runtime.asmcgocall

`shim.c:941` is `vkGetQueryPoolResults` with `VK_QUERY_RESULT_WAIT_BIT` — six
lines *past* the `vkWaitForFences` that was supposed to bound this. The fence
had already returned `VK_SUCCESS`. That is also why the shim's own 20-second
fence timeout never fired: the wait it would have bounded was already over.

## The kernel's side, which says what actually happened

`journalctl -k` carries one of these for every run that stalled, going back
through L8c-7's own sessions:

    amdgpu 0000:c6:00.0: ring gfx_0.0.0 timeout, signaled seq=115203707, emitted seq=115203710
    amdgpu 0000:c6:00.0:  Process llmbin pid 3038869 thread llmbin pid 3038881
    amdgpu 0000:c6:00.0: Starting gfx_0.0.0 ring reset
    amdgpu 0000:c6:00.0: Ring gfx_0.0.0 reset succeeded
    amdgpu 0000:c6:00.0: [drm] device wedged, but no recovery needed

So the job is killed by the ring watchdog and the ring is reset. A reset
signals the fences of the jobs it killed — which is the whole mechanism:

- `vkQueueSubmit` returns `VK_SUCCESS`.
- `vkWaitForFences` returns `VK_SUCCESS`, because the reset signalled it.
- the dispatches after the kill point never ran, so their timestamp slots are
  still `TIMESTAMP_NOT_READY`.
- `vkGetQueryPoolResults(WAIT_BIT)` polls those slots in userspace **forever**.

Replacing the wait flag with a deadline turns the hang into a sentence:

    shim: 2 of 1025 timestamp slots never became ready (first 1023, last 1024)
    llm: dispatches 0-1023: dispatch sequence failed: VkResult(2)

The first missing slot moves run to run (1023, then 980, then 977) because the
reset kills everything in flight, not one nominated dispatch. Dumping the
recorded sequence before the submit shows the grids around it are ordinary and
scale cleanly with the rows — `moe 10x952` at 2560 rows is `moe 10x1040` at
3072. **No dispatch is malformed. Nothing overflows.**

## The cliff is a duration

Rows in steps of 128 at 48 layers, 1024 dispatches a submit, timing the first
command buffer of the pass with its own GPU timestamps:

| rows | first submit | result |
|---:|---:|---|
| 2560 | 1.886 s | runs |
| 2688 | 1.964 s | runs |
| **2816** | **2.033 s** | **ring reset** |

And the control that separates time from everything it is confounded with —
same model, same rows, same dispatches, same grids, only the split changes:

| rows | dispatches a submit | submit | result |
|---:|---:|---:|---|
| 512 | 1273, one submit | 787 ms | runs |
| 2688 | 1273, one submit | ~2.43 s | **ring reset** |
| 2688 | 1024 + 249 | 1.964 s | runs |
| 3072 | 256 x 5 | 510-560 ms | runs, 1143.1 tok/s |

1273 dispatches in a single submit are fine at 512 rows and fatal at 2688; the
same 2688 rows are fine the moment the same work is split in two. **The bound
is the time one submit holds the ring, and it is 2.0 s.** The kernel constant
behind it was not identified — `lockup_timeout` is unset on the command line,
debugfs and `dmesg` both need root here — so the number is the measured one.

## The fix: chunk by time, not by count

`maxBatch = 1024` was a bound on the shim's query pool (`SHIM_QUERY_SLOTS`),
and the doc said so: "not a tuning knob". It happened to be a safe amount of
work for every pass in the vertical up to L9a, and stopped being one at 2816
rows.

A pass's cost is affine in the rows rather than proportional to them. Fitting
the whole-pass figures — 789.5 ms at 512 rows, 2342.4 at 2560, 2434.2 at 2688,
2687.5 at 3072, all over 1273 dispatches — gives

    322 us a dispatch  +  582 ns a dispatch-row

which reproduces all four to within 1.5%, and puts a 1024-dispatch submit at
2816 rows at **2.008 s** — the row count that is reset. `batchFor(rows)` fills
half of the 2 s cliff with that model. At one row it is the full 1024, so
**nothing about decode changes**; it starts biting at ~1700 rows and is 196 at
8192.

The extra fence waits cost ~40 us each and are invisible at every shape the
vertical measures: 2048 rows is 1037.2 tok/s against 1036.3 before, 2560 is
1088.8 against 1090.3, **decode is 24.48 tok/s against L8e's committed 24.66**
on the same widths and the same prompt — 0.7%, inside the 0.5-0.8% this vertical
reproduces to — with the same text and the same two command buffers a token, and
the hyper-connection block alone is still **4.22x** llama.cpp's.

`TestBatchForIsUnderTheWatchdog` checks the fit against the four measurements
and the bound against the cliff, because the constants are a fit in
milliseconds feeding a `time.Duration` and a factor of a thousand either way
still compiles — which it duly did, once.

Two things changed in the shim alongside it, both strictly better than what
they replace:

- every `vkGetQueryPoolResults` goes through `shim_query_results_deadline`,
  which polls to a ten-second deadline and then reports *which* slots never
  arrived. A killed submit is now an error with a cause, not a spin.
- `dispatchMulti`'s error is labelled `dispatch sequence` rather than
  `vkQueueSubmit`, which is where it was never coming from.

## The gates

`-graph` at 48 layers, the full prefill ladder, each row count its own submit
size:

| rows | ms | tok/s | vs llama.cpp's 391.42 | dispatches a submit |
|---:|---:|---:|---:|---:|
| 2048 | 1974.6 | 1037.2 | 2.65x | 660 |
| 2560 | 2351.2 | 1088.8 | 2.78x | 551 |
| 3072 | 2703.2 | 1136.4 | 2.90x | 473 |
| 3584 | 3092.6 | 1158.9 | 2.96x | 415 |
| 4096 | 3487.0 | **1174.6** | **3.00x** | 369 |
| 8192 | 6750.6 | **1213.5** | **3.10x** | 196 |

The 8192 row is at `-ctx 8192`, where the QSA selection is doing real work for
the first time in this vertical: attention is **14.7%** of the pass against
7.2% at 2560, and the MoE falls from 54.4% to 41.8%.

And the accuracy gate, which used to hang at 32 and 48 layers: `-ppl -ctx 4096`
completes at 48 on the checkpoint's own widths at **PPL = 3.9392 +/- 0.02209**,
over wikitext-2's 297,193 tokens in **72 chunks of 4096** scoring 2047 each,
in 7m24s. The same bank at n_ctx 2048 is 4.0289, so twice the context is
**-2.23%** with the selection live throughout — 2048 rows sits below the 2051
at which it is still the identity, 4096 is well past it. The instrument prints
its n_ctx 2048 comparison lines beside the result; those are **not** a
like-for-like delta across two chunkings, and here the only thing that moved
is the context.

**Prefill does not plateau.** llama.cpp's does — 388.60 at 2048, 392.95 at
8192 — and ours is still climbing at 8192. Every figure above 2560 rows is a
measurement that could not be taken before.

## What this leaves open

- **The kernel constant.** 2.0 s is measured, not read. Whether it is a
  per-ring `drm_sched` timeout, something the compositor's share of the same
  gfx ring imposes, or an APU-specific default is not settled, and it wants
  root to settle. The budget is half of it, which is margin enough either way.
- **The fit is this model's.** 322 us and 582 ns are the whole-model graph at
  48 layers. A 4-layer prefix pays a few more submits than it needs, which is
  ~40 us each and does not matter. A *different* model on this engine would
  want its own two numbers, or an adaptive bound measured from the previous
  submit — the recorder already gets each submit's GPU time back, so the hook
  is there if anything ever needs it.
- **L6a-4's "residency is free" is not the thing that was in question**, and
  comes out of P0 unmarked rather than confirmed: nothing here ever loaded it.
