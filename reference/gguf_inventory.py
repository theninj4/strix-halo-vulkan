"""Read a GGUF's headers and price it for this machine.

The point of this script is that **the header is 0.01% of the file**. A
sharded GGUF keeps its metadata in shard 1 and each shard's tensor table at
the start of that shard, so the complete inventory of a 111 GB checkpoint --
every tensor, its shape and its quantisation type -- can be had from one
small download plus three HTTP range requests. That is how LLM.md's tables
were produced before anything was downloaded in earnest.

    # the whole of shard 1 (metadata + tokenizer, ~11 MB) and 8 MB of each
    # of the rest, which is more than enough for the tensor tables
    B=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL
    N=Qwen3.8-Flash-Next-UD-Q4_K_XL
    curl -sL -o s1.gguf "$B/$N-00001-of-00004.gguf"
    for i in 2 3 4; do curl -sL -r 0-8388607 -o s$i.hdr "$B/$N-0000$i-of-00004.gguf"; done
    python3 reference/gguf_inventory.py s1.gguf s2.hdr s3.hdr s4.hdr

A truncated shard stops mid-table; the parser reports how far it got and
carries on, which is the expected path for the `.hdr` files.
"""

import collections
import json
import re
import struct
import sys

# ggml_type -> name. Only the ones a real checkpoint uses are listed.
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}

# (elements per block, bytes per block) -- ggml-common.h
BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8),
    "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "I64": (1, 8),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36), "IQ4_NL": (32, 18), "MXFP4": (32, 17),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ4_XS": (256, 136), "IQ3_S": (256, 110), "IQ3_XXS": (256, 98),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
    "IQ1_S": (256, 50), "IQ1_M": (256, 56),
}

# What this machine can do, measured: IDEAS §1.7 for the bus, and
# vulkaninfo for the heap. Both are quoted in LLM.md.
BUS_GB_S = 242.0
HEAP_DEVICE_LOCAL_GB = 89.97  # 83.79 GiB, RADV heap 1


class Reader:
    def __init__(self, buf):
        self.b, self.o = buf, 0

    def _u(self, fmt, n):
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += n
        return v

    def u32(self):
        return self._u("<I", 4)

    def u64(self):
        return self._u("<Q", 8)

    def string(self):
        n = self.u64()
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def value(self, t):
        if t == 0:
            return self._u("<B", 1)
        if t == 1:
            return self._u("<b", 1)
        if t == 2:
            return self._u("<H", 2)
        if t == 3:
            return self._u("<h", 2)
        if t == 4:
            return self.u32()
        if t == 5:
            return self._u("<i", 4)
        if t == 6:
            return self._u("<f", 4)
        if t == 7:
            return self._u("<B", 1) != 0
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u32(), self.u64()
            return [self.value(et) for _ in range(n)]
        if t == 10:
            return self.u64()
        if t == 11:
            return self._u("<q", 8)
        if t == 12:
            return self._u("<d", 8)
        raise ValueError(f"unknown gguf value type {t}")


def parse(path):
    """Return (kv, tensors). Tolerates a truncated tensor table."""
    with open(path, "rb") as f:
        buf = f.read()
    r = Reader(buf)
    if buf[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    r.o = 4
    r.u32()  # version
    n_tensor, n_kv = r.u64(), r.u64()
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.u32())
    tensors = []
    for _ in range(n_tensor):
        try:
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            ttype, _off = r.u32(), r.u64()
        except (struct.error, IndexError, UnicodeDecodeError):
            print(f"  {path}: tensor table truncated at {len(tensors)}/{n_tensor}",
                  file=sys.stderr)
            break
        tensors.append((name, dims, GGML_TYPE.get(ttype, f"?{ttype}")))
    return kv, tensors


def size_of(dims, ttype):
    n = 1
    for d in dims:
        n *= d
    elems, nbytes = BLOCK[ttype]
    return n, n // elems * nbytes


def group(name):
    """Bucket a qwen4exp tensor by the role it plays in the decode budget.

    Order matters: `hc_attn_*` contains "attn" and the DeltaNet layers' input
    projections are called `attn_qkv`/`attn_gate`, so the specific tests have
    to come before the generic "attn" one.
    """
    if name.startswith("per_layer_token_embd"):
        return "ngram_ple_table"
    if name.startswith("token_embd"):
        return "embed(lookup)"
    if name.startswith("output."):
        return "lm_head"
    if "exps" in name:
        return "moe_experts"
    if "shexp" in name:
        return "moe_shared"
    if "ffn_gate_inp" in name:
        return "moe_router"
    if "ssm" in name or "attn_qkv" in name or "attn_gate" in name:
        return "deltanet"
    if "indexer" in name:
        return "qsa_indexer"
    if "hc_" in name:
        return "hyper_conn"
    if "attn" in name:
        return "full_attn"
    if "ple" in name:
        return "ple_proj"
    return "norms/other"


# Groups that are gathered a row at a time rather than streamed whole, so
# they cost capacity but essentially no bandwidth.
GATHERED = {"ngram_ple_table", "embed(lookup)"}


def main(paths):
    tensors, kv = [], None
    for p in paths:
        k, t = parse(p)
        if kv is None:
            kv = k
        print(f"{p}: {len(t)} tensors, {len(k)} kv", file=sys.stderr)
        tensors += t

    g = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for name, dims, ttype in tensors:
        n, b = size_of(dims, ttype)
        row = g[group(name)]
        row[0] += n
        row[1] += b
        row[2][ttype] += b

    total_b = sum(v[1] for v in g.values())
    total_p = sum(v[0] for v in g.values())
    n_expert = int(kv.get("qwen4exp.expert_count", 512))
    n_used = int(kv.get("qwen4exp.expert_used_count", 10))

    print(f"\n{len(tensors)} tensors, {total_p / 1e9:.3f} B params, "
          f"{total_b / 1e9:.2f} GB, {total_b * 8 / total_p:.2f} bits/weight\n")
    print(f"{'group':17s} {'params B':>9s} {'GB':>7s} {'bits/w':>6s}  {'read':9s} quant mix (GB)")
    for name, (p, b, mix) in sorted(g.items(), key=lambda x: -x[1][1]):
        how = ("gather" if name in GATHERED else
               f"{n_used}/{n_expert}" if name == "moe_experts" else "every tok")
        blend = " ".join(f"{k}:{v / 1e9:.1f}" for k, v in mix.most_common())
        print(f"{name:17s} {p / 1e9:9.3f} {b / 1e9:7.2f} {b * 8 / p:6.2f}  {how:9s} {blend}")

    table = g["ngram_ple_table"][1]
    core = total_b - table
    dense = sum(b for n, (_, b, _) in g.items() if n not in GATHERED and n != "moe_experts")
    experts = g["moe_experts"][1] * n_used / n_expert
    per_token = dense + experts

    print(f"\nresident core (all but the n-gram table): {core / 1e9:.2f} GB")
    print(f"  against RADV heap 1 at {HEAP_DEVICE_LOCAL_GB} GB: "
          f"{HEAP_DEVICE_LOCAL_GB - core / 1e9:+.1f} GB")
    print(f"n-gram table, off-heap and mmap'd:        {table / 1e9:.2f} GB")
    print(f"\ndecode: dense {dense / 1e9:.3f} + experts {experts / 1e9:.3f} "
          f"= {per_token / 1e9:.3f} GB/token")
    print(f"  at {BUS_GB_S} GB/s (IDEAS §1.7) that is a {BUS_GB_S / (per_token / 1e9):.1f} tok/s ceiling")
    print(f"  dense is {100 * dense / per_token:.0f}% of it")

    json.dump([[n, d, t] for n, d, t in tensors], open("tensors.json", "w"))
    print("\ntensors.json written", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
