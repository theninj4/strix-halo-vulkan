"""Score a Kev suite partition with Kev's own fp32 code on the CPU (CLASSIFICATION.md K8).

The reference for `cmd/kev -compare`: the same rows `cmd/kev -out` writes (id, question, variant, source, task, type,
keys, label, p), from Kev's serving path (`model.probs`: the state once, then each question from its cache), loaded
as reference/dump_kev.py loads it (fp32, adapter merged, fitted temperature). Records are encoded under the serving
limits, as cmd/kev encodes them; every suite record was admitted under the smaller training context, so this is the
encoding `kev.benchmark` uses.

    .venv/bin/python reference/kev_suite_fp32.py --suite models/kev-suites/transfer-v4/development.jsonl --out fp32.jsonl

Rows already in --out are kept and skipped, so an interrupted run resumes. A full decision-v7 partition is ~2 h, so
--sample N scores a seeded random N records instead, and --also A,B adds every record where two row files (say the
fp16 and int8 banks) pick different answers, which are the rows the comparison is about.
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE  # noqa: E402


def label_index(m, y):
    """kev.benchmark.labels: a choice's label is its key, a noul's a bool, a score's the level index."""
    return m["keys"].index(y) if m["type"] == "choice" else int(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="models/kev-4b")
    ap.add_argument("--base", default="models/Qwen3.5-4B-Base")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=0, help="only the first n records")
    ap.add_argument("--sample", type=int, default=0, help="a seeded random sample of this many records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also", default="", help="two row files, comma-separated: add every record whose argmax they disagree on")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    ck = Checkpoint(a.run)
    if ck.meta.base_revision != "1001bb4d826a52d1f399e183466143f4da7b741b":
        raise SystemExit(f"head.pt pins {ck.meta.base_revision}, not the local base")
    ck.meta.base, ck.meta.base_revision = a.base, None
    tok, model = ck.load("cpu", LoadOptions())
    print(f"loaded: T={model.head.temperature}, dtype={model.dtype}, {len(done)} records already scored", flush=True)

    with open(a.suite) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if a.n:
        records = records[: a.n]
    if a.sample or a.also:
        keep = set()
        if a.sample:
            keep = {r["_meta"]["id"] for r in random.Random(a.seed).sample(records, min(a.sample, len(records)))}
        if a.also:
            x, y = ({(r["id"], r["question"]): r["p"] for r in map(json.loads, open(f))} for f in a.also.split(","))
            argmax = lambda p: max(range(len(p)), key=p.__getitem__)
            flips = {i for (i, q), p in x.items() if (i, q) in y and argmax(p) != argmax(y[(i, q)])}
            print(f"{len(flips)} records where {a.also} disagree", flush=True)
            keep |= flips
        records = [r for r in records if r["_meta"]["id"] in keep]
    done &= {r["_meta"]["id"] for r in records}
    start, n = time.time(), 0
    with open(a.out, "a") as out:
        for r in records:
            meta_r = r["_meta"]
            if meta_r["id"] in done:
                continue
            body = {"state": r["state"], "questions": {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                                                       for qid, q in r["questions"].items()}}
            rec, meta = to_record(SystemOneRequest(**body))
            enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            probs = model.probs(enc)
            rows = []
            for m, p, (qid, q) in zip(meta, probs, r["questions"].items()):
                rows.append({"id": meta_r["id"], "question": qid, "variant": meta_r["variant"], "source": meta_r["source"],
                             "task": q["src"], "type": m["type"], "keys": m["keys"], "label": label_index(m, q["label"]),
                             "p": [float(x) for x in p.tolist()]})
            out.write("".join(json.dumps(row) + "\n" for row in rows))   # one write a record, so a kill leaves whole records
            out.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - start
                print(f"{n + len(done)}/{len(records)} records, {el / n:.2f} s a record", flush=True)
    print(f"done: {n} records in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
