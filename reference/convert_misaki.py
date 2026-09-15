"""Export misaki's English pronunciation lexicon for the Go G2P (SPEECH.md T5).

`cmd/tts` takes phonemes, not text, and the thing between it and an HTTP
endpoint is grapheme-to-phoneme. Kokoro was trained on `misaki.en.G2P`'s
output, so that is the oracle — and the bulk of it is not an algorithm but two
dictionaries: `us_gold.json` (90201 entries, hand-checked) and
`us_silver.json` (93361, generated). Together they resolve **91% of the tokens
in real running text**, which is why the port starts here rather than with a
model.

The JSONs are shipped inside the misaki wheel, so a Go program cannot assume
they exist. This writes them as two plain text files that Go reads with a
`bufio.Scanner`, in the same spirit as `convert_kokoro.py`: the conversion is
the place where a Python-only artifact becomes something the engine can open,
and it checks itself on the way through.

The format is one entry per line, `word<TAB>value`, sorted by word:

    hello<TAB>hˈɛlO
    that<TAB>DEFAULT=ðæt|DT=ðˈæt
    read<TAB>ADJ=ɹˈɛd|DEFAULT=ɹˈid|VBD=ɹˈɛd|VBN=ɹˈɛd|VBP=ɹˈɛd

A value containing '=' is a tag-conditioned entry; an empty phoneme string
after a '=' is JSON `null`, which means *this* part of speech has no entry and
the lookup should fall through to the proper-noun path rather than to DEFAULT.
Tab, newline, '|' and '=' appear in no key and no value in either file, which
is checked below rather than assumed.

    .venv/bin/python reference/convert_misaki.py
"""

import argparse
import hashlib
import json
import os

# The separators the line format depends on being absent from the data.
RESERVED = "\t\n|="


def check_reserved(name, d):
    """Fail loudly if a key or value could be mistaken for a separator."""
    bad = set()
    for k, v in d.items():
        bad |= set(k) & set(RESERVED)
        if isinstance(v, str):
            bad |= set(v) & set(RESERVED)
        elif isinstance(v, dict):
            for tag, ps in v.items():
                bad |= set(tag) & set(RESERVED)
                if ps is not None:
                    bad |= set(ps) & set(RESERVED)
    if bad:
        raise SystemExit(f"{name}: reserved characters {sorted(bad)} in the data; "
                         "the line format would be ambiguous")


def encode(v):
    if isinstance(v, str):
        return v
    # Sorted so the file is reproducible and a diff means a data change.
    return "|".join(f"{tag}={'' if ps is None else ps}" for tag, ps in sorted(v.items()))


def decode(s):
    """The inverse, so the round trip can be checked here rather than in Go."""
    if "=" not in s:
        return s
    out = {}
    for part in s.split("|"):
        tag, _, ps = part.partition("=")
        out[tag] = ps if ps else None
    return out


def write(path, d):
    lines = []
    for word in sorted(d):
        lines.append(f"{word}\t{encode(d[word])}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    # Read it back and compare against the source, value for value: the same
    # cross-check convert_kokoro.py does on the tensors.
    back = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, value = line.rstrip("\n").partition("\t")
            back[word] = decode(value)
    if back != d:
        differing = [k for k in d if back.get(k) != d[k]][:5]
        raise SystemExit(f"{path}: round trip differs at {differing}")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return len(d), os.path.getsize(path), digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--misaki", default=None,
                    help="misaki package directory (default: the installed one)")
    ap.add_argument("-o", "--out", default="models/misaki")
    args = ap.parse_args()

    if args.misaki is None:
        import misaki
        args.misaki = os.path.dirname(misaki.__file__)
    data = os.path.join(args.misaki, "data")
    os.makedirs(args.out, exist_ok=True)

    manifest = {"source": data, "files": {}}
    for british in (False, True):
        prefix = "gb" if british else "us"
        for kind in ("gold", "silver"):
            src = os.path.join(data, f"{prefix}_{kind}.json")
            if not os.path.exists(src):
                continue
            d = json.load(open(src, encoding="utf-8"))
            check_reserved(src, d)
            name = f"{prefix}_{kind}.txt"
            n, size, digest = write(os.path.join(args.out, name), d)
            conditioned = sum(1 for v in d.values() if isinstance(v, dict))
            # An entry is only *ambiguous* if two tags give different phonemes;
            # the rest carry a single pronunciation under several keys, and a
            # Go port that ignored the tag would still get those right.
            ambiguous = sum(1 for v in d.values() if isinstance(v, dict)
                            and len({p for p in v.values() if p}) > 1)
            manifest["files"][name] = {
                "entries": n, "bytes": size, "sha256_16": digest,
                "tag_conditioned": conditioned, "tag_ambiguous": ambiguous,
            }
            print(f"{name}: {n} entries, {size/1e6:.2f} MB, {conditioned} tag-conditioned "
                  f"of which {ambiguous} actually differ by tag  [{digest}]")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
