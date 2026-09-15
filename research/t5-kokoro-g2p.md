# T5 — grapheme to phoneme for kokoro, in Go

*English text to the 178 IPA symbols `Kokoro-82M` reads, ported from
`misaki.en.G2P` — which is what the model was trained on. SPEECH.md's T5; the
code is `g2p/` (seven files), the oracles are `reference/convert_misaki.py`
and `reference/dump_g2p.py`, and it is validated by nine tests.*

**Headline: `go run ./cmd/tts -text 'In 2024 the team shipped 1,024 kernels
and spent $3.5 million, up 12%.'` speaks, with no Python on the path, and the
phonemes are character for character misaki's. The designed corpus is exact,
24 of 24 sentences; on 400 sentences nobody chose, 92.4% of phoneme words
agree.**

| | measure | result |
|---|---|---|
| the word path | 204 corpus tokens, phonemes and rating | **0 wrong** |
| `num2words` | 10115 integers x cardinal/ordinal/year, 11 decimals | **0 wrong** |
| `get_number` | 1760 cases, 45 digit strings x currency x head x flags | **0 wrong** |
| `Subtokenize` | 67 word shapes | **0 wrong** |
| the espeak fallback | 39 words, raw and rewritten | **0 wrong** |
| the homograph tagger | 1962 real occurrences, against spacy | **88.4%** (baseline 83.8%) |
| end to end, designed corpus | 24 sentences | **24 exact, 100% of words** |
| end to end, unseen prose | 400 sentences | 68.8% exact, **92.4% of words** |

## The measurement that decided the design

misaki runs **spacy**, and porting a neural tagger would have been larger than
everything else in `SPEECH.md`. So the first step was not code. Over 54174
tokens of this repository's own prose — real running English, and a hard case
because a tenth of it is acronyms, so the numbers are floors:

    gold dictionary          44229      81.64%
    silver dictionary         3745       6.91%
    the three suffix rules    1456       2.69%
    resolved                             91.24%
    unresolved                4744       8.76%

**91% of English is a table lookup.** And of the 90201 gold entries only 790
are tag-conditioned, of which 671 actually give different phonemes for
different tags — **1.74% of running-text tokens**. Half of what *looks* like
tagging is not: misaki's `'None'` key is selected by `ctx.future_vowel is
None`, by whether a vowel follows, which accounts for another 1.83% and needs
no tagger at all. `that`, `this`, `by`, `has`, `be`, `would` and `there` carry
a stressed form for the end of a phrase and an unstressed one for the middle.

So the port is: the dictionary and its rules (exact), the numbers (exact), a
tokenizer that reproduces spacy's *splitting* but not its tagging, twelve
rules where the tagging mattered, and espeak for the rest.

## Test the oracle's decomposition, not just its output

The single most useful decision was to have the dump record what
`Lexicon.get_word` and `get_number` made of each token **in isolation**, along
with the two context values each was given — and, because misaki does not keep
the context, to recover it by replaying the reverse walk over the finished
tokens.

That turned an all-or-nothing port into a pure function with 204 test cases.
It also caught the one thing a naive dump got wrong immediately: with an empty
context `will` comes out `wˈɪl`, and it should be `wɪl`.

The same principle, one level down: **dump the cross product, not the corpus.**
The corpus reaches 21 number tokens and `get_number` has eight branches keyed
on digit count, a leading zero, a decimal point, `is_head` and currency. 45
digit strings against every combination of the other three is 1760 cases and
costs nothing to generate. And a table catches what reading does not — the
subtoken port looked correct and split `3.50` into `3.5` and `0`, because
`(?:\d?[,.]?\d)+` matches a run's last digit with *both* its optional parts
empty and a hand-written loop does not unless it tries the alternatives in the
engine's order. Three of 67 words failed; nothing else would have found it.

## The tokenizer is where the port stops being a translation

Four things the end-to-end measurement found that no component test could:

**A hyphen inside a word is not punctuation.** `well-known` subtokenizes to
`well | - | known`, and that hyphen is inside a word while the period after
`dog` is beside one — the same character class, opposite treatment. Only the
tokenizer knows which is which, so it marks it. Guessing later splits the run,
which loses both the merge loop's chance to look the whole word up and the
stress rule that demotes half of what it finds.

**Ask the dictionary whether a period belongs to the word.** It carries `Dr.`
as an entry, so `l.gold[chunk]` answers the abbreviation question with no list
to maintain. `U.S.A.` is the one pattern left over.

**`prespace`.** `12%` is two pieces the tokenizer glued together and has to
come apart again as "twelve percent"; misaki decides that by whether the run
mixed letters with digits or carried a space.

**A currency symbol reaches forwards.** It is spent on the *last* CD token of
the run that follows, so `$3.5 million` is millions of dollars — which only
works if the scale words are tagged CD, as spacy tags them.

## The homograph tagger: measure the rule, do not reason about it

Twelve rules over one word either side, and the metric is the **pronunciation**
rather than the tag: two tags that select the same entry are equally right, and
most do. That is why the no-tagger baseline is already 83.8% — DEFAULT *is* the
frequent form, by construction.

The first attempt scored **79.2%, worse than no tagger at all**, and all of the
damage was one word. `that` is 445 of the 1962 occurrences and the only one
whose distinction is not noun-against-verb: spacy calls it DT when it
determines a noun phrase and WDT or IN when it opens a relative clause, and
only DT is stressed. The signal turned out to be entirely on the left — a
relative `that` follows the noun it modifies, a determiner follows a boundary
or a function word:

    always not-DT (the baseline)                67.0%
    DT iff prev is punctuation or sentence start 73.5%
    DT iff prev is punct/start/function word     83.1%   <- chosen
      ... and next is not a verb                 74.6%
    DT iff prev is punct/start/preposition       82.2%

**The obvious refinement is wrong and costs 8.5 points.** "A determiner cannot
be followed by a verb" sounds like syntax and fails, because "that is" is
overwhelmingly a determiner in running prose. It was only found by trying it.

The other lesson is the opposite shape. "A content word in front means a
subject, so the verb is tensed" takes `read` from 63 to 70 of 104 — and
applied generally it is a disaster, because "a memory fragment" becomes the
verb and `fragment` alone is 79 occurrences. Restricted to the four entries
that carry a VBD key, it is a clean gain. **A rule that helps one word is not
a rule until it is measured on the others.**

## espeak is a library here, not a binary

The plan this stage started with said "an `espeak-ng` shell-out". There is no
`espeak-ng` binary on this machine — only the shared library bundled inside
misaki's wheel, at a path one Python version away from being wrong. So the
binding is **cgo plus `dlopen`**, which also means `go build` works on a
machine with no espeak at all and a caller that gets an error simply has no
fallback, exactly as the engine behaved before T5d.

Three layers stack in the fallback and the dump records all three, so a
disagreement says which one moved:

1. **espeak** with `phonememode = IPA | tie | ('͡' << 8)`. The tie is what makes
   misaki's rewrite table expressible: without it, the `eɪ` of one diphthong
   and the `eɪ` spanning a syllable boundary are the same two characters.
2. **phonemizer**, which is two things — the tie becomes `^`, and a word
   containing punctuation is phonemized in pieces with the mark put back, so
   `10:30` is `tˈɛn:θˈɜːɾi` rather than two words.
3. **misaki's 25 rules**, whose order is load-bearing: `e^ɪ` before `e`, `ə^l`
   before `l`, `ʲo` before the bare `ʲ` that is deleted.

The American tail is not symmetrical with the British one. `ɜː` becomes `ɜɹ` —
the rhotic vowel espeak spells as a length mark — and then *every* remaining
length mark is dropped, because kokoro's American vocabulary has no `ː` in it.

## What the two end-to-end numbers mean

The designed corpus is 24 sentences in which every branch of misaki's English
G2P fires, and it is **exact**. That says the pieces are right and wired
together right; it does not say they generalise, because the port was aimed at
it.

The 400 sentences lifted out of this repository's prose are the other
measurement: **68.8% of sentences and 92.4% of phoneme words**, on text that is
a tenth acronyms with `vs.`, `i.e.`, `W8A8` and section numbers in it. Reading
the differences, some are the tagger's 12%, some are spacy tokenizing a
fragment oddly — and in at least one case misaki reads `vs.` as `vˈiz` where
this port says `vˈɜɹsəs`, which is not a difference anyone should want fixed.

## What this leaves

`cmd/tts` takes text. The remaining work in `SPEECH.md` is **T6**: the phoneme
side is 214 ms of a 252 ms utterance, 60% of it six bidirectional LSTMs whose
steps are GEMVs at M = 1 — S8's problem in a different model.

For T5 itself the open items are small and each has its number attached: the
tagger's 11.6%, the tokenizer's disagreements with spacy on fragments, and
`Model.Style` indexing the voice pack by the **character** count of the phoneme
string minus what fell outside the vocabulary — which is how `KPipeline` does
it, is not the token count, and means a G2P that emits a different number of
characters picks a different voice row.
