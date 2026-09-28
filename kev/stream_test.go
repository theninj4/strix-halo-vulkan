package kev

import (
	"fmt"
	"strings"
	"testing"
)

// TestGPUStreamMatchesWhole is K9's gate: a request whose state runs in
// chunks -- alone, with other requests' passes between its chunks, and
// sharing passes with them -- answers with the bits of its state in one pass,
// and so do the requests around it. The state is then cached, and a hit
// answers the same.
func TestGPUStreamMatchesWhole(t *testing.T) {
	g := loadGPU(t)
	e := loadEncoder(t)
	fx := loadFixtures(t)
	long, _ := ParseRequest(fx[2].Request)
	rec, _ := ToRecord(long)
	short, _ := ParseRequest(fx[0].Request)
	srec, _ := ToRecord(short)
	rec.State = strings.Repeat(rec.State+"\n\n", 2)
	rec.Questions = append(rec.Questions, srec.Questions...)
	enc, err := e.Encode(rec, MaxState, MaxRow)
	if err != nil {
		t.Fatal(err)
	}
	var shorts []*Encoding
	for i := range 4 {
		r := srec
		r.State = fmt.Sprintf("Ticket %d. %s", i, srec.State)
		se, err := e.Encode(r, MaxState, MaxRow)
		if err != nil {
			t.Fatal(err)
		}
		shorts = append(shorts, se)
	}
	t.Logf("stream: %d tokens, state %d, %d questions", len(enc.IDs), enc.StateLen, len(enc.Decide))

	g.ClearCache()
	whole, _, err := g.Probs(enc)
	if err != nil {
		t.Fatal(err)
	}
	alone := make([][][]float64, len(shorts))
	for i, se := range shorts {
		g.ClearCache()
		if alone[i], _, err = g.Probs(se); err != nil {
			t.Fatal(err)
		}
	}

	defer func(c int) { g.ChunkRows = c }(g.ChunkRows)
	for _, chunk := range []int{128, 192} {
		for _, mode := range []string{"alone", "between", "sharing"} {
			g.ChunkRows = chunk
			g.ClearCache()
			if err := g.BeginStream(enc); err != nil {
				t.Fatal(err)
			}
			var got *Pass
			steps, next := 0, 0
			checkShorts := func(ps []*Pass, idx []int) {
				for j, p := range ps {
					i := idx[j]
					pr := g.headProbs([]*Encoding{shorts[i]}, []*Pass{p})[0]
					if same, worst := sameProbs(alone[i], pr); !same {
						t.Errorf("chunk %d, %s: short %d moved by %.2e", chunk, mode, i, worst)
					}
				}
			}
			for got == nil {
				var encs []*Encoding
				var idx []int
				if mode != "alone" {
					i := next % len(shorts)
					next++
					encs, idx = []*Encoding{shorts[i]}, []int{i}
				}
				if mode == "between" {
					_, ps, err := g.Step(false, encs)
					if err != nil {
						t.Fatal(err)
					}
					checkShorts(ps, idx)
					encs, idx = nil, nil
				}
				g.ClearCache() // the shorts stay cold, so each run writes its state
				done, ps, err := g.Step(true, encs)
				if err != nil {
					t.Fatal(err)
				}
				checkShorts(ps, idx)
				got = done
				steps++
			}
			pr := g.headProbs([]*Encoding{enc}, []*Pass{got})[0]
			same, worst := sameProbs(whole, pr)
			t.Logf("chunk %d, %s: %d steps, identical %v (max |dp| %.2e)", chunk, mode, steps, same, worst)
			if !same {
				t.Errorf("chunk %d, %s: the stream moved by %.2e", chunk, mode, worst)
			}
			if g.Streaming() {
				t.Fatal("still streaming after its last step")
			}
		}
	}
	// The finished stream's state is cached: a hit answers the same.
	p, pass, err := g.Probs(enc)
	if err != nil {
		t.Fatal(err)
	}
	if !pass.CacheHit {
		t.Error("the stream's state was not cached")
	}
	if same, worst := sameProbs(whole, p); !same {
		t.Errorf("a cache hit on the stream's state moved by %.2e", worst)
	}
}
