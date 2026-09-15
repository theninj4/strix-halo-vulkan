"""Dump a reference run of misaki's English G2P, for stage T5 of SPEECH.md.

`cmd/tts` takes phonemes, not text. Kokoro was trained on `misaki.en.G2P`'s
output over a 178-symbol IPA vocabulary, so misaki is the oracle for the Go
port exactly as `KModel` was for the vocoder — and, like that one, this walks
the middle rather than the ends: every token's tag, its phonemes and the path
that produced them, so a partial port can be checked against the part it
implements.

Two things this dump exists to settle, both of which decide how much of misaki
has to be reproduced at all:

  * **How much is dictionary and how much is algorithm.** Measured over the
    repository's own prose — 54 k tokens of real running English, and hostile
    to a lexicon because it is full of acronyms — 91% of tokens resolve from
    `us_gold`/`us_silver` directly or through the three stemming rules. The
    algorithm is what covers the rest: numbers, currency, ordinals, acronyms
    and an espeak fallback.

  * **How much a part-of-speech tagger is worth.** misaki runs spacy, which is
    a neural tagger, and porting one would dwarf the rest of T5. But only 790
    of 90201 gold entries are tag-conditioned and only 671 of those actually
    differ between tags. The corpus survey below splits the tag-conditioned
    hits into the ones a tagger decides and the ones decided by **local
    context** — misaki's `'None'` key is selected by `ctx.future_vowel is
    None`, i.e. by whether a vowel follows, not by any tag — so the manifest
    reports what a tagger-free port would actually get wrong.

The corpus is recorded verbatim in the manifest, so the Go side needs neither
misaki nor spacy nor espeak installed to run its tests, which is the same rule
`dump_kokoro.py` follows for its phoneme string.

Self-checks: every sentence's per-token phonemes are re-joined with their
whitespace and compared against the whole-string output, and every phoneme
produced is checked against kokoro's own vocabulary.

    .venv/bin/python reference/dump_g2p.py
"""

import argparse
import collections
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

from misaki import en

# Sentences chosen so that every branch of misaki's English G2P fires at least
# once, and labelled with what each is for. They are the Go port's unit tests.
CORPUS = [
    ("plain", "The quick brown fox jumps over the lazy dog."),
    ("plain", "She sells sea shells by the sea shore."),
    ("the-before-vowel", "The apple and the pear are on the table in the autumn."),
    ("homograph-read", "He read the book yesterday; she will read it tomorrow."),
    ("homograph-noun-verb", "They record a record; we present a present; I object to the object."),
    ("homograph-live", "They live to see a live broadcast."),
    ("contractions", "I can't believe it's already been years since we haven't met, don't you think?"),
    ("possessive", "Anthropic's model wrote the compiler's output to the user's home directory."),
    ("numbers-cardinal", "There are 1024 lanes, 37 waves and 5 arenas."),
    ("numbers-large", "The bank moved 1,024,000 dollars and counted 1000000 coins."),
    ("numbers-year", "In 2024 and in 1999, and again in 1900 and 2001."),
    ("numbers-decimal", "The kernel reached 38.8 TFLOP per second at 0.5 volts."),
    ("numbers-ordinal", "The 1st, 2nd, 3rd and 4th of June, and the 21st of May."),
    ("currency", "It cost $3.50, or £2, or €1.99 in total."),
    ("percent-symbols", "Throughput rose 12% and 50% & fell @ dawn + dusk."),
    ("time", "The meeting is at 10:30 and ends at 2:15."),
    ("acronym", "The GPU ran GLSL and SPIR-V through the CPU and the DRAM."),
    ("proper-nouns", "Nvidia and Anthropic met in Zurich with Kraftwerk."),
    ("oov", "The florbulent gnorpsile quixotified the wambulator."),
    ("punctuation", "Wait — really? Yes: absolutely, without question... truly!"),
    ("quotes", 'She said "hello there" and then left.'),
    ("hyphenation", "A well-known, state-of-the-art, twenty-one-year-old approach."),
    ("stress-context", "That is this, and by then it has been there through all of these."),
    ("mixed", "In 2024, Dr. Smith's team reported $3.5 million in revenue, up 12% year over year."),
]


def replay_context(g2p, tokens):
    """The TokenContext each token saw, recovered by walking backwards.

    misaki does not keep it: the main loop carries one `ctx` through a reverse
    walk and overwrites it each step. But `token_context` is a pure function of
    the *next* token's final phonemes, so replaying the same walk over the
    finished tokens reproduces exactly what each lookup was given — which is
    what makes `get_word` a pure function the Go port can be tested as.
    """
    from misaki import en

    ctx = en.TokenContext()
    out = [None] * len(tokens)
    for i in reversed(range(len(tokens))):
        out[i] = ctx
        ctx = en.G2P.token_context(ctx, tokens[i].phonemes, tokens[i])
    return out


def word_path(g2p, token, ctx):
    """What `Lexicon.get_word` alone makes of this token, and the context it saw.

    T5b is exactly this function — the special cases, the dictionary, the three
    stemming rules and the stress rules — and nothing else: not the tokenizer,
    not `num2words`, not espeak. So it is recorded separately from the token's
    final phonemes, along with the two inputs it is not a pure function
    without, so that the Go port can be checked as a pure function.

    `stress` is misaki's own: None for an all-lowercase word, 0.5 for a
    capitalised one and 2 for an all-caps one. `future_vowel` is whether the
    next spoken phoneme is a vowel, None at a sentence end or before
    punctuation — which is what selects a lexicon entry's 'None' key, and the
    reason 1.8% of tokens *look* like they need a part-of-speech tagger and do
    not.
    """
    import unicodedata
    from misaki import en

    word = token.text.replace(chr(8216), "'").replace(chr(8217), "'")
    word = unicodedata.normalize("NFKC", word)
    word = "".join(en.Lexicon.numeric_if_needed(c) for c in word)
    stress = None if word == word.lower() else g2p.lexicon.cap_stresses[int(word == word.upper())]
    try:
        ps, rating = g2p.lexicon.get_word(word, token.tag, stress, ctx)
    except Exception:
        ps, rating = None, None
    if ps is not None:
        # The same two substitutions the main loop makes at the end.
        ps = ps.replace("\u027e", "T").replace("\u0294", "t")
    # The number path, recorded in the same isolated way and with the three
    # inputs it takes from the tokenizer rather than from the text: the
    # currency symbol that preceded it, whether it is the first subtoken of its
    # group, and the flags `preprocess` set.
    num_ps, num_rating = None, None
    if ps is None and en.Lexicon.is_number(word, token._.is_head):
        try:
            num_ps, num_rating = g2p.lexicon.get_number(
                word, token._.currency, token._.is_head, token._.num_flags or "")
        except Exception:
            num_ps, num_rating = None, None
        if num_ps is not None:
            num_ps = num_ps.replace("\u027e", "T").replace("\u0294", "t")
    return {
        "word": word,
        "stress": stress,
        "future_vowel": ctx.future_vowel,
        "future_to": ctx.future_to,
        "word_ps": ps,
        "word_rating": rating,
        "currency": token._.currency,
        "is_head": bool(token._.is_head),
        "num_flags": token._.num_flags or "",
        "num_ps": num_ps,
        "num_rating": num_rating,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="reference/out/g2p")
    ap.add_argument("--british", action="store_true")
    ap.add_argument("--no-fallback", action="store_true",
                    help="omit the espeak fallback, so out-of-vocabulary words come back as None")
    ap.add_argument("--survey", nargs="*", default=["README.md", "SPEECH.md", "TODO.md",
                                                    "IDEAS.md", "GOALS.md"],
                    help="markdown files to measure lexicon coverage over")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fallback = None
    if not args.no_fallback:
        from misaki import espeak
        fallback = espeak.EspeakFallback(british=args.british)
    g2p = en.G2P(trf=False, british=args.british, fallback=fallback)

    manifest = {
        "british": args.british,
        "fallback": "espeak" if fallback is not None else None,
        "sentences": [],
    }

    # kokoro's own vocabulary, so a symbol misaki can emit but the model cannot
    # read is caught here rather than as silence in a WAV.
    try:
        from kokoro.model import KModel
        vocab = set(KModel(disable_complex=True).vocab) if False else None
    except Exception:
        vocab = None
    if vocab is None:
        # The vocabulary is also in the converted checkpoint's config.
        cfg = "models/Kokoro-82M/config.json"
        vocab = set(json.load(open(cfg))["vocab"]) if os.path.exists(cfg) else None

    joined_gap = 0
    outside = collections.Counter()
    word_path_n = 0
    num_path_n = 0
    for kind, text in CORPUS:
        ps, tokens = g2p(text)
        ctxs = replay_context(g2p, tokens)
        toks = [{
            "text": t.text,
            "tag": t.tag,
            "whitespace": t.whitespace,
            "phonemes": t.phonemes,
            # misaki writes the confidence to `_.rating` on some paths and to a
            # plain attribute on others, so read both: 4 is the gold lexicon,
            # 3 silver or a spelled-out proper noun, 2 the espeak fallback.
            "rating": t._.rating if t._.rating is not None else getattr(t, "rating", None),
            **word_path(g2p, t, c),
        } for t, c in zip(tokens, ctxs)]
        word_path_n += sum(1 for t in toks if t["word_ps"] is not None)
        num_path_n += sum(1 for t in toks if t["num_ps"] is not None)
        # Self-check: the per-token phonemes, rejoined, are the whole string.
        rejoined = "".join((t["phonemes"] or "") + t["whitespace"] for t in toks).strip()
        if rejoined != ps.strip():
            joined_gap += 1
        if vocab:
            outside.update(c for c in ps if c not in vocab and c != " ")
        manifest["sentences"].append({
            "kind": kind, "text": text, "phonemes": ps, "tokens": toks,
        })
        print(f"{kind:22s} {text[:44]!r:48s} -> {ps[:58]!r}")

    manifest["self_checks"] = {
        "token_rejoin_mismatches": joined_gap,
        "phonemes_outside_kokoro_vocab": dict(outside),
        "vocab_checked": vocab is not None,
        "word_path_tokens": word_path_n,
        "number_path_tokens": num_path_n,
    }
    print(f"\nself-check: {joined_gap} of {len(CORPUS)} sentences disagree with their own tokens; "
          f"{len(outside)} symbols outside kokoro's vocabulary {dict(outside)}")

    manifest["numbers"] = dump_numbers(os.path.join(args.out, "numbers.txt"))
    manifest["wild"] = dump_wild(g2p, os.path.join(args.out, "wild.txt"), args.survey)
    manifest["homographs"] = dump_homographs(
        g2p, os.path.join(args.out, "homographs.txt"), args.survey)
    manifest["espeak"] = dump_espeak(os.path.join(args.out, "espeak.txt"), args.british)
    manifest["subtokens"] = dump_subtokens(os.path.join(args.out, "subtokens.txt"))
    manifest["number_cases"] = dump_number_cases(
        g2p, os.path.join(args.out, "number_cases.txt"))
    manifest["survey"] = survey(g2p, args.survey)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)
    print(f"wrote {args.out}/manifest.json")


def dump_numbers(path):
    """An exhaustive table of `num2words`, which misaki's number path is built on.

    misaki spells a number by calling `num2words` and then splitting the result
    on anything that is not a lowercase letter, so what a Go port has to
    reproduce is the *word sequence* — but the separators matter too, because
    one of them is the word "and", which `num_flags` can keep.

    Three forms, because misaki uses three: cardinal for an ordinary number,
    ordinal for `1st`, and year for a bare four-digit one. Every integer from 0
    to 10000 is covered exhaustively rather than sampled, because the join
    rules change at 100 and at every scale boundary and a sample would miss
    them; above that the interesting values are the boundaries themselves and
    their neighbours.

    Written as a tab-separated table rather than into the manifest: it is 10 k
    lines, and it is data for one test rather than context for a reader.
    """
    from num2words import num2words as n2w

    values = list(range(0, 10001))
    for scale in (10 ** 5, 10 ** 6, 10 ** 9, 10 ** 12, 10 ** 15):
        for d in (-1, 0, 1, 2, 50, 100, 101, 999, 1000, 1001, 100000):
            for sign in (1, -1):
                v = sign * (scale + d)
                if v not in values and abs(v) < 10 ** 16:
                    values.append(v)
    values += [-1, -21, -100, -1234]

    lines = []
    for v in values:
        year = ""
        if 0 <= v <= 9999:
            year = n2w(v, to="year")
        ordinal = n2w(v, to="ordinal") if v >= 0 else ""
        lines.append(f"{v}\t{n2w(v)}\t{ordinal}\t{year}\n")

    # Decimals, which take a different branch: num2words spells the fraction
    # digit by digit rather than as a number.
    decimals = ["0.5", "3.5", "1.99", "38.8", "0.05", "12.25", "100.001",
                "0.0", "7.0", "1234.5678", "0.125"]
    for d in decimals:
        lines.append(f"{d}\t{n2w(float(d))}\t\t\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote {path}: {len(values)} integers and {len(decimals)} decimals")
    return {"file": os.path.basename(path), "integers": len(values), "decimals": len(decimals)}


# Words chosen to reach every alternative of misaki's subtoken regex, which is
# the one piece of its tokenizer that is not spacy's.
SUBTOKEN_WORDS = [
    "hello", "well-known", "state-of-the-art", "twenty-one-year-old",
    "don't", "it's", "Anthropic's", "users'", "'cause", "'tis'",
    "can't", "o'clock", "rock'n'roll", "''quoted''", "\u2018single\u2019",
    "GPU", "GPUDevice", "camelCase", "XMLHttpRequest", "iPhone", "eBay",
    "1024", "1,024", "1,024,000", "3.50", "0.5", "-5", "38.8", "1.2.3",
    "21st", "1990s", "10:30", "12%", "$3.50", "hello@example.com",
    "a_b", "a__b", "snake_case_name", "SPIR-V", "co-op", "re-enter",
    "U.S.A.", "Dr.", "e.g.", "...", "—", "\u201chello\u201d", "(paren)",
    "M\u00fcnchen", "Krak\u00f3w", "caf\u00e9", "na\u00efve",
    "", "-", "'", "''", "2", "a2b", "x2y",
    # The uppercase-before-Upper-Lower alternative, which nothing above reaches.
    "XYz", "ABCde", "aBCd", "HTTPServer", "IOError", "macOS", "AI", "AIx",
]


# The words the fallback exists for: proper nouns, invented words, accented
# spellings, and the two shapes that are not words at all.
ESPEAK_WORDS = [
    "Kraftwerk", "Nvidia", "Anthropic", "Zurich", "Vulkan", "Strix", "gfx1151",
    "florbulent", "gnorpsile", "quixotified", "wambulator", "snorgle",
    "caf\u00e9", "na\u00efve", "M\u00fcnchen", "Krak\u00f3w", "Z\u00fcrich",
    "10:30", "2:15", "SPIR-V", "well-known", "e-mail",
    "GEMM", "GEMV", "TFLOP", "coopmat", "RDNA", "iGPU", "safetensors",
    "hello", "the", "read", "record", "twenty", "dog",
    "Schadenfreude", "Weltanschauung", "rendezvous", "colonel",
]


def dump_wild(g2p, path, paths):
    """misaki's output for sentences nobody chose.

    The 24-sentence corpus above is a *designed* test: every branch fires, and
    a port that passes it has been aimed at it. This is the other measurement —
    real sentences lifted out of the repository's own prose, with whatever
    vocabulary, punctuation and numbers they happen to contain — and it is the
    one that says whether the tokenizer and the homograph rules generalise or
    were fitted.
    """
    text = []
    for p in paths:
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read()
        t = re.sub(r"```.*?```", " ", t, flags=re.S)
        t = re.sub(r"`[^`]*`", " ", t)
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"[|*#\[\]]", " ", t)
        text.append(t)
    blob = " ".join(" ".join(text).split())
    sentences, cur = [], []
    for word in blob.split(" "):
        cur.append(word)
        if word.endswith((".", "!", "?")) and len(cur) >= 6:
            s = " ".join(cur)
            if 40 <= len(s) <= 300 and sum(c.isalpha() for c in s) > len(s) // 2:
                sentences.append(s)
            cur = []
        if len(cur) > 60:
            cur = []
    sentences = sentences[:400]
    rows = []
    for s in sentences:
        ps, _ = g2p(s)
        rows.append(s.replace("\t", " ") + "\t" + ps.replace("\t", " ") + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"wrote {path}: {len(rows)} sentences")
    return {"file": os.path.basename(path), "sentences": len(rows)}


def dump_homographs(g2p, path, paths):
    """Every occurrence of a tag-ambiguous word in the survey corpus, with the
    tag spacy gave it and the two words around it (SPEECH.md T5d).

    This is the only thing a part-of-speech tagger is needed for: 671 of 90201
    gold entries give different phonemes for different tags, and T5a measured
    that they are 1.74% of running-text tokens. The question a Go port has to
    answer is not "how accurate is spacy" but "how often does a *rule* on the
    neighbouring words give the same pronunciation", and that can only be
    asked against a table of real occurrences.

    The tag is recorded twice: as spacy gave it, and collapsed onto the coarse
    key the dictionary is indexed by, since two tags that select the same entry
    are equally right.
    """
    gold = g2p.lexicon.golds
    ambiguous = {k for k, v in gold.items()
                 if isinstance(v, dict) and len({p for p in v.values() if p}) > 1}

    text = []
    for p in paths:
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read()
        t = re.sub(r"```.*?```", " ", t, flags=re.S)
        t = re.sub(r"`[^`]*`", " ", t)
        t = re.sub(r"https?://\S+", " ", t)
        text.append(t)
    rows = []
    for para in "\n".join(text).split("\n\n"):
        para = " ".join(para.split())
        if not para or len(para) > 4000:
            continue
        doc = g2p.nlp(para)
        toks = [t for t in doc]
        for i, t in enumerate(toks):
            if t.text.lower() not in ambiguous:
                continue
            prev = toks[i - 1].text if i > 0 else ""
            nxt = toks[i + 1].text if i + 1 < len(toks) else ""
            rows.append(f"{t.text}\t{t.tag_}\t{en.Lexicon.get_parent_tag(t.tag_)}\t{prev}\t{nxt}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"wrote {path}: {len(rows)} occurrences of {len(set(r.split(chr(9))[0].lower() for r in rows))} words")
    return {"file": os.path.basename(path), "occurrences": len(rows),
            "ambiguous_entries": len(ambiguous)}


def dump_espeak(path, british):
    """misaki's `EspeakFallback`, which is what a word outside the dictionary
    gets (SPEECH.md T5d).

    Three layers stack here and the Go port has to reproduce all three: espeak
    itself, phonemizer's wrapper around it (a tie character and punctuation
    preservation), and misaki's 25-rule rewrite of espeak's IPA into kokoro's
    178-symbol alphabet. So the table records espeak's raw output as well as
    misaki's, and the Go test can say *which* layer disagrees when one does.
    """
    import ctypes

    import espeakng_loader
    from misaki import espeak as mespeak
    from misaki.token import MToken

    lib = ctypes.CDLL(espeakng_loader.get_library_path())
    lib.espeak_Initialize.restype = ctypes.c_int
    lib.espeak_Initialize(0x02, 0, espeakng_loader.get_data_path().encode(), 0)
    lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
    lib.espeak_SetVoiceByName(b"gmw/en-GB" if british else b"gmw/en-US")
    lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
    lib.espeak_TextToPhonemes.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
    mode = 0x02 | (0x01 << 7) | (ord("\u0361") << 8)

    def raw(t):
        ptr = ctypes.pointer(ctypes.c_char_p(t.encode()))
        out = []
        while ptr.contents.value is not None:
            r = lib.espeak_TextToPhonemes(ptr, 1, mode)
            if r:
                out.append(r.decode())
        return " ".join(out)

    fb = mespeak.EspeakFallback(british=british)
    rows = []
    for w in ESPEAK_WORDS:
        ps, rating = fb(MToken(text=w, tag="NNP", whitespace=""))
        rows.append(f"{w}\t{raw(w)}\t{fb.backend.phonemize([w])[0].strip()}\t{ps}\t{rating}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"wrote {path}: {len(rows)} words")
    return {"file": os.path.basename(path), "words": len(rows),
            "library": espeakng_loader.get_library_path(),
            "data": espeakng_loader.get_data_path(),
            "voice": "gmw/en-GB" if british else "gmw/en-US",
            "phoneme_mode": mode}


def dump_subtokens(path):
    """misaki's subtoken regex, applied to every shape it was written for.

    It is the one part of the tokenizer that is misaki's rather than spacy's,
    and it is a single nine-alternative regex with two lookaheads — which Go's
    RE2 cannot express, so the port is a hand-written scanner and needs a table
    to be held to. Camel case, internal apostrophes, comma-grouped digits and
    runs of hyphens each take a different alternative.
    """
    rows = []
    for w in SUBTOKEN_WORDS:
        parts = en.subtokenize(w)
        rows.append(f"{w}\t{'|'.join(parts)}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"wrote {path}: {len(rows)} words")
    return {"file": os.path.basename(path), "words": len(rows)}


def dump_number_cases(g2p, path):
    """`get_number` over a cross product, not only over what the corpus says.

    The corpus reaches 21 number tokens, and `get_number` has eight branches
    whose conditions turn on the digit count, a leading zero, a decimal point,
    whether the token heads its group and whether a currency symbol preceded
    it. Twenty-one tokens cannot pin that down, and the branches are exactly
    where a port goes quietly wrong — a four-digit number is a *year* unless it
    is an amount, and "0.5" and ".5" and "3.50" take three different paths.

    So: every interesting digit string against every currency, both values of
    is_head, and the three flag combinations the preprocessor can set.
    """
    words = [
        "0", "5", "12", "37", "100", "101", "110", "709", "700", "1024",
        "1990", "1900", "1999", "2000", "2001", "2024", "2100", "10000",
        "1,024", "1,024,000", "1000000", "0.5", ".5", "3.5", "3.50", "38.8",
        "1.99", "0.05", "12.25", "100.001", "1.2.3", "-5", "-1234",
        "1st", "2nd", "3rd", "4th", "21st", "101st", "1990s", "1000's",
        "007", "0123", "42ing", "3'd",
    ]
    rows = []
    for word in words:
        for currency in (None, "$", "£", "€"):
            for is_head in (True, False):
                for flags in ("", "&", "n", "a", "&a"):
                    if not en.Lexicon.is_number(word, is_head):
                        continue
                    try:
                        ps, rating = g2p.lexicon.get_number(word, currency, is_head, flags)
                    except Exception:
                        continue
                    if ps is None:
                        continue
                    ps = ps.replace("\u027e", "T").replace("\u0294", "t")
                    rows.append(f"{word}\t{currency or ''}\t{int(is_head)}\t{flags}\t{ps}\t{rating}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"wrote {path}: {len(rows)} cases over {len(words)} digit strings")
    return {"file": os.path.basename(path), "cases": len(rows), "words": len(words)}


def survey(g2p, paths):
    """Measure what a Go port would have to implement, over real running text.

    Markdown, with code fences and inline code stripped: what is left is prose,
    and this repository's prose is a hard case for a lexicon because a tenth of
    it is acronyms. The point of measuring on something hostile is that the
    coverage number is then a floor.
    """
    text = []
    for p in paths:
        if not os.path.exists(p):
            continue
        s = open(p, encoding="utf-8").read()
        s = re.sub(r"```.*?```", " ", s, flags=re.S)
        s = re.sub(r"`[^`]*`", " ", s)
        s = re.sub(r"https?://\S+", " ", s)
        text.append(s)
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", "\n".join(text))
    if not words:
        return {}

    lex = g2p.lexicon
    gold, silver = lex.golds, lex.silvers
    ambiguous = {k for k, v in gold.items()
                 if isinstance(v, dict) and len({p for p in v.values() if p}) > 1}

    path = collections.Counter()
    oov = collections.Counter()
    for w in words:
        lw = w.lower()
        if w in gold or lw in gold:
            path["gold"] += 1
        elif w in silver or lw in silver:
            path["silver"] += 1
        else:
            for suffix, n in (("'s", 2), ("es", 2), ("ing", 3), ("ed", 2), ("s", 1)):
                if lw.endswith(suffix) and (lw[:-n] in gold or lw[:-n] in silver):
                    path["stem" + suffix] += 1
                    break
            else:
                path["unresolved"] += 1
                oov[lw] += 1

    # Of the tag-conditioned hits, how many would a tagger-free port get wrong?
    # misaki's 'None' key is chosen by local context (whether a vowel follows),
    # not by a tag, so an entry whose only non-DEFAULT keys are 'None' costs
    # nothing. The rest are the homographs a tagger is actually for.
    tagged = collections.Counter()
    for w in words:
        lw = w.lower()
        v = gold.get(lw) if isinstance(gold.get(lw), dict) else gold.get(w)
        if not isinstance(v, dict) or lw not in ambiguous and w not in ambiguous:
            continue
        keys = {k for k in v if k not in ("DEFAULT", "None")}
        if keys:
            tagged["needs_a_tagger"] += 1
            tagged["word:" + lw] += 1
        else:
            tagged["context_only"] += 1

    n = len(words)
    out = {
        "files": [p for p in paths if os.path.exists(p)],
        "tokens": n,
        "types": len(set(w.lower() for w in words)),
        "paths": {k: v for k, v in path.most_common()},
        "paths_pct": {k: round(100 * v / n, 2) for k, v in path.most_common()},
        "resolved_pct": round(100 * (n - path["unresolved"]) / n, 2),
        "tagger": {
            "needs_a_tagger": tagged["needs_a_tagger"],
            "needs_a_tagger_pct": round(100 * tagged["needs_a_tagger"] / n, 2),
            "context_only": tagged["context_only"],
            "distinct_words": sorted(k[5:] for k in tagged if k.startswith("word:")),
        },
        "top_unresolved": [w for w, _ in oov.most_common(40)],
    }
    print(f"\nsurvey over {n} tokens of {', '.join(out['files'])}:")
    for k, v in out["paths"].items():
        print(f"  {k:12s} {v:7d}  {out['paths_pct'][k]:5.2f}%")
    print(f"  {'resolved':12s} {'':7s}  {out['resolved_pct']:5.2f}%")
    t = out["tagger"]
    print(f"  a tagger decides {t['needs_a_tagger']} tokens ({t['needs_a_tagger_pct']:.2f}%), "
          f"{len(t['distinct_words'])} distinct words; {t['context_only']} more are context, not tags")
    return out


if __name__ == "__main__":
    main()
