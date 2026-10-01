#!/usr/bin/env python3
"""Price fixed and confidence-stopped draft depths from a SPEC_PROBE file.

Each probed round (depth-2 loop, one probe draft deeper) gives the three
drafts' probabilities q1..q3 and A, the run of drafts the trunk agreed with
(0..3; 3 is seen only where the round kept every row). A policy picks the
round's depth D from the q's as it drafts; the round then yields 1+min(A,D)
tokens for pass(D+1) + drafts steps. Rounds are treated as samples (the
position shift between depths is ignored).
"""
import re, sys, itertools

# P20h (research/p20-llamacpp-mtp.md §13): python3 p20h_simdepth.py results/p20h_probe_d2.txt [four-row pass in steps]
PASS = {1: 1.0, 2: 1.29, 3: 1.52, 4: float(sys.argv[2]) if len(sys.argv) > 2 else 1.75}
DRAFT = 0.093   # a draft step, in trunk steps (§9)
OVER = 0.015    # host per round, calibrated on §10's depth-1/2 rates

rounds = []
for li, line in enumerate(open(sys.argv[1])):
    head, d, q, o = [x.strip() for x in line.split("|")]
    run, at, keep = map(int, head.split())
    if li == 0:
        continue  # the warm-up's one round
    d = list(map(int, d.strip("[]").split()))
    q = list(map(float, q.strip("[]").split()))
    o = list(map(int, o.strip("[]").split()))
    # The probe (d[2]) is checked only where the round kept every row and
    # the trunk named the token after (o has 3 entries); a stop (EOG/budget)
    # may cut o short, which then reads as a rejection.
    a = 0
    if len(o) >= 1 and d[0] == o[0]:
        a = 1
        if len(o) >= 2 and d[1] == o[1]:
            a = 2
            if len(o) >= 3 and d[2] == o[2]:
                a = 3
    rounds.append((q, a))

n = len(rounds)
print(f"{n} rounds")
for k in (1, 2, 3):
    num = sum(1 for q, a in rounds if a >= k)
    den = sum(1 for q, a in rounds if a >= k - 1)
    print(f"a{k}|prev = {num}/{den} = {num/den:.3f}")


def price(policy):
    tok = cost = 0.0
    for q, a in rounds:
        D, drafted = policy(q)
        tok += 1 + min(a, D)
        cost += PASS[D + 1] + drafted * DRAFT + OVER
    return tok / cost, tok / n


def fixed(D):
    return lambda q: (D, D)


def stop(t, maxd, keep_low):
    """Draft until a draft's probability is under t[k] (or maxd). keep_low:
    the low draft still goes into the pass (llama.cpp's p_min drops it)."""
    def p(q):
        for k in range(maxd):
            if q[k] < t[k]:
                return (k + 1 if keep_low else k), k + 1
        return maxd, maxd
    return p


for D in (1, 2, 3):
    r, t = price(fixed(D))
    print(f"fixed depth {D}: {r:.3f}x  {t:.2f} tok/round")

best = []
grid = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
for maxd in (2, 3):
    for keep_low in (False, True):
        for ts in itertools.product(grid, repeat=maxd):
            r, t = price(stop(ts + (0,) * (3 - maxd), maxd, keep_low))
            best.append((r, t, maxd, keep_low, ts))
best.sort(reverse=True)
for maxd in (2, 3):
    for keep_low in (False, True):
        bs = [b for b in best if b[2] == maxd and b[3] == keep_low][:3]
        for b in bs:
            print(f"{b[0]:.3f}x {b[1]:.2f} tok/round  maxd {b[2]} keep_low {b[3]} thresholds {b[4]}")
