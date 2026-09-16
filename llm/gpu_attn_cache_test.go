package llm

import (
	"fmt"
	"testing"
)

// L7a: the KV cache, and the one claim that makes decode possible at all.
//
// Everything above this file runs a full-attention layer as a *fresh
// sequence*: token t is cell t, its position is t, and the key and value
// planes are the batch's own padded token rows. Decode is the opposite — one
// token at a time, at cell `past`, attending over every cell before it — and
// the only way that is the same model is if splitting a prompt into chunks
// changes nothing.
//
// So that is the gate, and it is an equality rather than a tolerance. A
// chunked run recomputes nothing: the key and value cells a previous chunk
// wrote are read back as they were written, the indexer's pooled blocks are
// carried over, and the flash-attention key loop walks the same absolute
// blocks in the same order. Every dispatch is therefore the *same arithmetic
// on the same values*, and the output has to come back bit for bit. A
// tolerance here would pass a cache that was subtly off by a position.
//
// The fixture is the 4 k one, because at 256 cells a chunk boundary can only
// fall in one place and the selection never bites. At 4096 tokens the
// schedules below put boundaries inside a pooled indexer block, inside a
// 16-row fragment tile, and — the decode case — at every single token.

// runChunks runs the layer over `in` in the given chunk sizes, starting from a
// fresh sequence, and returns the layer's output for every token.
func runChunks(t *testing.T, g *AttnGPU, in []float32, nEmbd int, chunks []int) []float32 {
	t.Helper()
	g.Reset()
	out := make([]float32, 0, len(in))
	past := 0
	for _, n := range chunks {
		if err := g.SetPast(past); err != nil {
			t.Fatal(err)
		}
		if err := g.Upload(in[past*nEmbd:(past+n)*nEmbd], n); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(0); err != nil {
			t.Fatalf("chunk of %d at cell %d: %v", n, past, err)
		}
		out = append(out, g.Out()...)
		past += n
	}
	return out
}

// even cuts n into chunks of size k, with whatever is left over last.
func even(n, k int) []int {
	var out []int
	for i := 0; i < n; i += k {
		out = append(out, minInt(k, n-i))
	}
	return out
}

// TestAttnGPUCacheIsAChunkSplit is L7a's gate.
func TestAttnGPUCacheIsAChunkSplit(t *testing.T) {
	g, tr, c, nTok, done := attnGPU4k(t)
	defer done()
	src, err := tr.Get("hc_mixed-3", 0)
	if err != nil {
		t.Fatal(err)
	}
	in := src.Vals

	// Both runs on one pinned plan. The rungs are chosen from the *run's*
	// length by default, and a rung is a different tiling — so leaving that
	// on would be asking whether two different kernels agree bit for bit,
	// which is not the question this test is about.
	attn, gemm, outGemm := g.Plan()
	if err := g.SetPlan(attn, gemm, outGemm); err != nil {
		t.Fatal(err)
	}
	t.Logf("plan pinned at %s / %s / %s, sparse %v", attn, gemm, outGemm, g.Sparse())

	want := runChunks(t, g, in, c.NEmbd, []int{nTok})

	for _, tc := range []struct {
		name   string
		chunks []int
	}{
		{"512 at a time", even(nTok, 512)},
		{"64 at a time, inside a fragment tile", even(nTok, 64)},
		{"7 at a time, crossing every block boundary", even(nTok, 7)},
		{"a prompt and then one token at a time", append([]int{nTok - 33}, even(33, 1)...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := runChunks(t, g, in, c.NEmbd, tc.chunks)
			if len(got) != len(want) {
				t.Fatalf("%d values over %d chunks, want %d", len(got), len(tc.chunks), len(want))
			}
			bad, first := 0, -1
			for i := range got {
				if got[i] != want[i] {
					bad++
					if first < 0 {
						first = i
					}
				}
			}
			if bad != 0 {
				t.Errorf("%d of %d values differ, first at token %d component %d: %v against %v",
					bad, len(got), first/c.NEmbd, first%c.NEmbd, got[first], want[first])
				return
			}
			t.Logf("%d chunks, %d tokens: identical to the last place", len(tc.chunks), nTok)
		})
	}
}

// TestAttnGPUCacheKeepsPooledBlocks is the negative control for the one thing
// the chunk split above cannot see.
//
// A pooled indexer block is final as soon as its `ratio` cells exist, so a
// continuing run rebuilds only the blocks its own tokens completed and reads
// every earlier one out of the table. If the table were *not* carried over —
// if a chunk rebuilt only its own blocks and left the rest of the table as
// whatever the previous layer or the previous sequence wrote — the equality
// above would still hold whenever each chunk happened to complete every block
// it needed. This asserts the range directly: it is [past/ratio,
// (past+T)/ratio) on a continuing run and the whole table on a fresh one.
func TestAttnGPUCacheKeepsPooledBlocks(t *testing.T) {
	g, _, c, _, done := attnGPU4k(t)
	defer done()
	for _, tc := range []struct{ past, tokens, lo, hi int }{
		{0, 4096, 0, g.NBlocks()}, // fresh: the whole table, dead blocks and all
		{0, 7, 0, g.NBlocks()},    // fresh and short: still the whole table
		{4000, 96, 1000, 1024},    // the dead blocks become real
		{4000, 1, 1000, 1000},     // one token, no block completed
		{4003, 1, 1000, 1001},     // one token, and it finishes block 1000
		{7, 1, 1, 2},              // a chunk boundary inside block 1
	} {
		name := fmt.Sprintf("%d tokens at cell %d", tc.tokens, tc.past)
		if err := g.SetPast(tc.past); err != nil {
			t.Fatal(err)
		}
		if err := g.Resize(tc.tokens); err != nil {
			t.Fatal(err)
		}
		lo, hi := g.blockRange()
		if lo != tc.lo || hi != tc.hi {
			t.Errorf("%s: blocks [%d, %d), want [%d, %d)", name, lo, hi, tc.lo, tc.hi)
			continue
		}
		// Every block the run must own: complete by the end of it, and not
		// complete before it started.
		for b := 0; b < g.NBlocks(); b++ {
			done := (b+1)*c.Ratio <= tc.past+tc.tokens
			was := (b+1)*c.Ratio <= tc.past
			need := done && !was && tc.past != 0
			if got := b >= lo && b < hi; tc.past != 0 && got != need {
				t.Errorf("%s: block %d dispatched %v, completed-by-this-run %v", name, b, got, need)
			}
		}
	}
}
