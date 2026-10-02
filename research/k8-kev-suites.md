# K8 — accuracy against Kev's card (2026-09-25)

*The K8 stage, broken out of `CLASSIFICATION.md` when it moved to `research/` on 2026-10-02; the vertical's frame, decisions and handoff are in [`classification-vertical.md`](classification-vertical.md).*


`cmd/kev -suite` runs one of Kev's frozen suites and scores it as
`kev.benchmark` and `kev.metrics` do: argmax, Brier and ECE over the clean,
knowable rows, raw Brier from the served p (p^T renormalised is
softmax(z)), and the unknowable share at p ≥ 0.9. `-compare` pairs two row
files. `reference/kev_suite_fp32.py` writes the same rows from Kev's own
code in fp32 on the CPU (`model.probs`, ~1.3 s a transfer-v4 record, ~5 s a
decision-v7 one; it resumes an interrupted file). The suites are copied
from Kev's repo at `9fdf054` into `models/kev-suites/{transfer-v4,
decision-v7, transfer-v9}` (development and test). Rows land in
`reference/out/kev/suites/` (gitignored).

**The oracle reproduces Kev's card to the digit**, so the card's numbers
are fp32 numbers and the comparison below is like for like:

| | Kev's card | Kev fp32, here | fp16 bank | int8 bank (served) |
|---|---|---|---|---|
| transfer-v4 dev, accuracy (656) | 0.817 | **0.8171** | 0.8155 | 0.8140 |
| transfer-v4 dev, Brier served / raw | 0.243 / 0.269 | **0.2430 / 0.2691** | 0.2430 / 0.2691 | 0.2422 / 0.2681 |
| transfer-v4 dev, ECE | 0.042 | **0.0416** | 0.0425 | 0.0396 |
| transfer-v4 test, accuracy / Brier (656) | 0.838 / 0.224 | **0.8384 / 0.2242** | 0.8384 / 0.2241 | 0.8384 / 0.2238 |
| decision-v7 dev (1,264) | 0.873 | sampled | **0.8726** | 0.8687 |
| decision-v7 test (1,200) | 0.865 | – | **0.8650** | 0.8642 |
| transfer-v9 dev, MMLU-Pro (200) | 0.565 | sampled | **0.565** | 0.560 |
| transfer-v9 dev, unknowable at ≥ 0.9 (110) | 0.00 | sampled | 0.00 | 0.00 |

decision-v7 dev's raw Brier on fp16, **0.2037**, is also the raw Brier
`head.json` records for the temperature fit on those same 1,264 rows. The
card's transfer-v9 "0.490" and "buried 0.67" are the *previous*
checkpoint's; the current one publishes only MMLU-Pro and the unknowable
share. (Here buried is 0.700 on both banks; transfer-v9 dev overall 0.7734
fp16, 0.7715 int8.)

**Row by row against Kev's fp32** (`-compare`):

| | fp16 bank | int8 bank |
|---|---|---|
| transfer-v4 dev, argmax agreement | **763/764** | 761/764 |
| transfer-v4 dev, max / mean max \|dp\| | 0.0030 / 0.00015 | 0.027 / 0.0024 |
| transfer-v4 test, argmax agreement | **764/764** | 764/764 |
| transfer-v4 test, max / mean max \|dp\| | 0.0011 / 0.00012 | 0.030 / 0.0023 |
| decision-v7 dev, sampled: argmax agreement | **399/400** | 395/400 |
| transfer-v9 dev, sampled: argmax agreement | **155/155** | 148/155 |

The last two rows are a sample, not a full run (decision-v7 is ~1.7 s a
record in fp32): a seeded random 150 records (`--sample 150`), every
record where the fp16 and int8 banks disagree (`--also`, 6 and 7 of them),
and on decision-v7 the first 183 records of an interrupted full run. The
flips are chosen on purpose, so int8's agreement there is not a rate.

**fp16's disagreements are ties in fp32**: the top two options are within
0.000–0.002 of each other, on all four partitions. The fp16 bank matches
fp32 to about Kev's own isolation-gate tolerance (1e-3), and it lands on
every published accuracy exactly or within one question. **int8's are
not all ties**: fp32 margins ≤ 0.002 on transfer-v4, but up to 0.016 on
decision-v7 (sst5, yelp) and 0.032 on transfer-v9's 10-way MMLU-Pro,
where three of its four flips land on option 8 (too few to call a
pattern).

**What int8 costs.** It is a real quantisation, so its flips are not
symmetric noise. Against fp16 over the five partitions with an fp16
control it is **−9 questions of 4,822 (−0.19 pp)**: transfer-v4 dev −1
(−2 against fp32), test 0, decision-v7 dev −5 (1462/1468 argmax agreement, all six flips at
p ≈ 0.49 against 0.49), decision-v7 test −1, transfer-v9 dev −2. Calibration
does not move (Brier within 0.001). That is smaller than the gap between
Kev's bf16 server and its fp32 (~0.03 in p). It buys 1.2–1.4x on short
passes (K7.1); `-kev-fp16` is the exact option and costs 3.3 GB more.
