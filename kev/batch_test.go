package kev

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"strix-halo-vulkan/safetensors"
)

// TestGPUBatchMatchesSingle is K7.5's gate: the four fixtures as one pass
// answer as each does alone -- cold, then with every state cached, then with
// half of them cached -- and the batch really is one pass.
func TestGPUBatchMatchesSingle(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	var encs []*Encoding
	for _, fx := range loadFixtures(t) {
		encs = append(encs, encodeFixture(t, e, fx.Request))
	}
	alone := make([][][]float64, len(encs))
	for i, enc := range encs {
		g.ClearCache()
		p, _, err := g.Probs(enc)
		if err != nil {
			t.Fatal(err)
		}
		alone[i] = p
	}
	check := func(name string) {
		probs, ps, err := g.ProbsBatch(encs)
		if err != nil {
			t.Fatal(err)
		}
		worst, same, hits := 0.0, true, 0
		for i := range encs {
			s, w := sameProbs(alone[i], probs[i])
			same, worst = same && s, math.Max(worst, w)
			if ps[i].CacheHit {
				hits++
			}
			if ps[i].Batch != len(encs) {
				t.Errorf("%s: request %d ran in a batch of %d", name, i, ps[i].Batch)
			}
		}
		t.Logf("%s: %d cached, identical %v, max |dp| %.2e, batch GPU %v", name, hits, same, worst, ps[0].GPU)
		// Exact with either attention: a request's answers do not depend on
		// its partners in the pass, or on what ran before it (K7.7).
		if !same {
			t.Errorf("%s: max |dp| %.2e, not identical", name, worst)
		}
	}
	g.ClearCache()
	check("cold")
	check("every state cached")
	g.ClearCache()
	for _, enc := range encs[:2] {
		if _, _, err := g.Probs(enc); err != nil {
			t.Fatal(err)
		}
	}
	check("half cached")
}

// TestGPUBatchThroughput: eight distinct short requests one at a time, then
// as one batch, cold each time. Wall clock.
func TestGPUBatchThroughput(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	base, _ := ParseRequest(loadFixtures(t)[0].Request)
	rec, _ := ToRecord(base)
	for _, n := range []int{2, 4, 8} {
		var encs []*Encoding
		for i := 0; i < n; i++ {
			r := rec
			r.State = fmt.Sprintf("Ticket %d. %s", i, rec.State)
			enc, err := e.Encode(r, MaxState, MaxRow)
			if err != nil {
				t.Fatal(err)
			}
			encs = append(encs, enc)
		}
		best := func(f func()) time.Duration {
			b := time.Duration(1 << 62)
			for range 3 {
				g.ClearCache()
				start := time.Now()
				f()
				b = min(b, time.Since(start))
			}
			return b
		}
		seq := best(func() {
			for _, enc := range encs {
				if _, _, err := g.Probs(enc); err != nil {
					t.Fatal(err)
				}
			}
		})
		bat := best(func() {
			if _, _, err := g.ProbsBatch(encs); err != nil {
				t.Fatal(err)
			}
		})
		t.Logf("%d requests (%d tokens each): one at a time %.1f ms, batched %.1f ms (%.2fx, %.1f req/s)",
			n, len(encs[0].IDs), float64(seq.Microseconds())/1000, float64(bat.Microseconds())/1000,
			float64(seq)/float64(bat), float64(n)/bat.Seconds())
	}
}

// TestGPUStaleCacheDoesNotLeak fills every attention layer's K and V cache
// with junk before each run and requires the answers to be the same bits as
// over a zeroed cache: cold, batched, and from the prefix cache. Before K7.7
// the V padding after a segment -- masked, but inside a key block a row sees
// -- moved kev_attn_wmma's sums by an fp16 ulp, so a request's answers
// depended on what had run before it.
func TestGPUStaleCacheDoesNotLeak(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	var encs []*Encoding
	for _, fx := range loadFixtures(t) {
		encs = append(encs, encodeFixture(t, e, fx.Request))
	}
	rng := rand.New(rand.NewSource(7))
	block := make([]uint16, 64*kvRow)
	fill := func(junk bool) {
		for i := range block {
			block[i] = 0
			if junk {
				block[i] = safetensors.F32ToF16(float32(rng.NormFloat64() * 4))
			}
		}
		for i := range g.w {
			if !g.w[i].attn {
				continue
			}
			for _, off := range []uint32{g.w[i].kc, g.w[i].vc} {
				for c := 0; c < g.cells+64; c += 64 {
					g.hbuf.WriteUint16At(int(off)+c*kvRow, block[:min(64, g.cells+64-c)*kvRow])
				}
			}
		}
	}
	probs := func(junk, hit bool) [][][]float64 {
		g.ClearCache()
		if hit {
			if _, _, err := g.ProbsBatch(encs); err != nil {
				t.Fatal(err)
			}
		}
		fill(junk)
		p, _, err := g.ProbsBatch(encs)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	var alone [][][]float64
	for _, enc := range encs {
		g.ClearCache()
		fill(false)
		p, _, err := g.Probs(enc)
		if err != nil {
			t.Fatal(err)
		}
		alone = append(alone, p)
	}
	for _, c := range []struct {
		name      string
		junk, hit bool
	}{{"batched, zeroed cache", false, false}, {"batched, junk cache", true, false}, {"batched from the prefix cache, junk", true, true}} {
		got := probs(c.junk, c.hit)
		for i := range encs {
			if same, worst := sameProbs(alone[i], got[i]); !same {
				t.Errorf("%s: request %d moved by %.2e", c.name, i, worst)
			}
		}
	}
	for i, enc := range encs {
		g.ClearCache()
		fill(true)
		p, _, err := g.Probs(enc)
		if err != nil {
			t.Fatal(err)
		}
		if same, worst := sameProbs(alone[i], p); !same {
			t.Errorf("alone, junk cache: request %d moved by %.2e", i, worst)
		}
	}
}
