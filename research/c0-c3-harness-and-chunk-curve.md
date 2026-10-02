# C0 + C3 — the harness and the chunk curve (measured 2026-09-23)

*The C0 + C3 section, broken out of `CONCURRENCY.md` when it moved to `research/` on 2026-10-02; the plan, the stage table and the log are in [`concurrency.md`](concurrency.md).*


**The harness is `cmd/loadgen`.** It sends two background agents at t=0
(a 7.3k/7.8k-token wikitext document each, "summarise it at length",
`max_tokens` 768, thinking on) and four interactive voice commands at 2, 9,
25 and 40 s. Each voice command is a Home-Assistant-shaped system prompt of 40
devices (**1.94k tokens**), a short utterance, `reasoning_effort: none` and
`max_tokens` 32. The voice commands at 2 s and 9 s land while the agents are
prefilling, and the ones at 25 s and 40 s land while they are decoding.
Before each concurrent phase it runs every stream alone as the same-hour
control. Every arm below ran the concurrent phase twice, and **the two runs
agree within 0.03 s on every TTFT and 0.3 tok/s on every rate** (the
baseline's within 0.07 s). The server was the 48-layer model at
`-llm-ctx 65536 -llm-batch 8192`, with `ai.service` stopped. The sweep script
restarts the server per arm from one binary.

| arm | voice TTFT at 2 s / 9 s (in prefill) | at 25 s / 40 s (in decode) | agent decode, each | agent prefill alone (7338 tok) | aggregate |
|---|---|---|---|---|---|
| alone (control) | 1.84 / 1.79 | 1.80 / 1.79 | 34.8 | — | — |
| **1 slot (before C2)** | **27.4 / 22.7** | 9.0 / 24.1 | 35.4, but agent1 waits 40.6 s for its first token | 6.32 s (8192 chunk) | 24.8 tok/s |
| 3 slots, chunk 512 | 2.05 / 1.83 | 1.90 / 1.79 | 16.8 | 11.71 s (**+85%**) | 21.3 |
| 3 slots, chunk 1024 | 2.09 / 2.01 | 1.81 / 1.79 | 16.1 | 9.10 s (+44%) | 22.9 |
| **3 slots, chunk 2048** (the default) | 3.41 / 2.44 | 1.81 / 1.79 | 16.2 / 15.7 | 7.29 s (+15%) | 24.2 |
| 3 slots, chunk 4096 | 3.11 / 4.80 | 1.82 / 1.81 | 16.2 / 14.6 | 6.43 s (+2%) | 24.7 |
| 3 slots, chunk 8192 | 5.74 / 7.35 | 1.83 / 1.79 | 13.1 / 15.4 | 6.32 s | 24.8 |

(TTFTs in seconds and rates in tok/s, all as the client sees them.)

What it says:

1. **C2 does what it was for.** A voice command that arrives while the agents
   are decoding gets its solo TTFT and its solo rate (35.4 tok/s against 34.4
   alone) at every chunk size. The agents stall for those ~2 s and nothing
   else happens to them. One that arrives during an agent's prefill waits for
   the chunk on the device and no longer. Before C2 it waited for a whole
   generation, 9-27 s.
2. **Switching slots costs nothing measurable.** The two agents' decode
   windows (~48.8 s for 768 tokens each, minus ~4.4 s of voice turns inside
   them) come to 34.6 tok/s combined, which is the solo rate. Aggregate
   throughput is the same as one slot's to within the prefill-chunk cost, as
   the design said: C2 buys fairness, not tokens. C5 buys tokens.
3. **A prefill chunk has a fixed cost of ~0.39 s**, and it is the MoE bank.
   Fitting chunk time = c0 + n·c1 over the solo prefills gives c0 ≈ 0.39 s
   and c1 ≈ 0.80 ms a token. 0.39 s is one read of the ~76 GB expert bank at
   ~200 GB/s: every chunk of 512 or more rows touches essentially every
   expert. So a chunk under ~2k is mostly bank reading. At 512 the cost is
   +85% on background prefill, and the aggregate falls 14%.
4. **The worst voice wait is one chunk's duration**, not what the two sample
   arrivals happened to hit: ~0.8 s at 512, **~1.3 s at 1024, ~2.0 s at
   2048**, ~3.3 s at 4096 and ~6.3 s at 8192.

**Decision: the default stays 2048.** Going to 1024 saves ~0.7 s of worst-case
voice wait and costs every background prefill 25% (whenever there is more than
one slot, whether or not a voice command ever comes). The larger part of a
voice command's TTFT today is its **own** 1.8 s prefill of a system prompt
that is identical every time, and C4 removes that. Revisit the chunk after C4,
when the chunk wait will be most of what is left. A different lever is the
chunk's fixed cost: if an MoE pass over 1024 rows read less than the whole
bank, small chunks would be cheap, but at 1024 rows × 8 experts of 256 it
does not.
