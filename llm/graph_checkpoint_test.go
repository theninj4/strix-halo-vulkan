package llm

// CONCURRENCY.md C4: taking a slot back to a checkpoint.
//
//	go test ./llm/ -v -run TestGraphCheckpointIsThePrefill   # four layers, ~9 GB

import (
	"fmt"
	"testing"
)

// TestGraphCheckpointIsThePrefill is C4's gate. A prompt prefix is
// checkpointed, the slot runs on into a different continuation (a prompt
// tail and some decode steps, which move every recurrence and complete
// pooled indexer blocks past the checkpoint), and it is restored and run
// into the real tail. That must give **bit for bit** the logits of running
// prefix and tail with nothing in between.
//
// The prefix is 2600 tokens in a 4096-cell cache, which puts the indexer's
// selection past its width, so the pooled blocks decide which cells are read.
// The detour (600 tokens and 12 steps) runs further than the real
// continuation (300 and 12), so blocks it completed are still there, stale,
// when the real one decodes. That is the case in which Checkpoint's claim
// that the pooled table needs no restore could fail, and it does not. The
// sequence runs in slot 1 while slot 0 runs something else between the
// checkpoint and the restore, so the checkpoint is also checked to be the
// slot's and not the graph's.
//
// Each control leaves one part of the restore out, and each must move the
// logits. Otherwise the gate cannot tell that part of the restore from a
// no-op.
func TestGraphCheckpointIsThePrefill(t *testing.T) {
	const (
		layers = 4
		chunk  = 1024
		nKV    = 4096
		prefix = 2600
		steps  = 12
	)
	m, tr := fixtures4k(t)
	all, _, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < 3600 {
		t.Skipf("the 4k trace has %d tokens", len(all))
	}
	pre, tail, other := all[:prefix], all[prefix:prefix+300], all[3000:3600]

	dev, done := newTestDevice(t)
	t.Cleanup(done)
	g, err := NewGraph(dev, m, GraphOpts{MaxTokens: chunk, NKV: nKV, Layers: layers, Slots: 2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)
	if !g.attn.sparse {
		t.Fatalf("a %d-cell cache is not sparse, so the pooled blocks would not be under test", nKV)
	}

	clone := func(l []float32) []float32 { return append([]float32(nil), l...) }
	argmax := func(l []float32) int32 {
		id := int32(0)
		for j := range l {
			if l[j] > l[id] {
				id = int32(j)
			}
		}
		return id
	}
	use := func(s int) {
		if err := g.UseSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	// feed runs ids through the live slot in `chunk`-wide passes, the first
	// one fresh when asked, and returns the last pass's logits.
	feed := func(ids []int32, fresh bool) []float32 {
		var logits []float32
		for at := 0; at < len(ids); at += chunk {
			c := ids[at:min(at+chunk, len(ids))]
			var err error
			if fresh && at == 0 {
				logits, _, err = g.Forward(c)
			} else {
				logits, _, err = g.Extend(c)
			}
			if err != nil {
				t.Fatalf("%d tokens at %d: %v", len(c), g.Past(), err)
			}
		}
		return logits
	}
	// finish runs the tail and then `steps` greedy decode steps, or the
	// steps named when forced is non-nil, and returns every logit row.
	finish := func(forced []int32) ([]int32, [][]float32) {
		logits := feed(tail, false)
		rows := [][]float32{clone(logits)}
		var toks []int32
		for i := range steps {
			id := argmax(logits)
			if forced != nil {
				id = forced[i]
			}
			toks = append(toks, id)
			var err error
			if logits, _, err = g.Extend([]int32{id}); err != nil {
				t.Fatal(err)
			}
			rows = append(rows, clone(logits))
		}
		return toks, rows
	}

	// The reference: prefix then tail, nothing between. Twice, because the
	// first run over fresh arenas is not bit-repeatable against later ones
	// (TestPrerecordedDecodeIsTheRecordedDecode); slot 0 is warmed too.
	use(0)
	feed(other, true)
	use(1)
	feed(pre, true)
	finish(nil)
	feed(pre, true)
	toks, want := finish(nil)

	// restore is Graph.Restore with a part optionally left out, for the
	// controls. The full one is g.Restore itself.
	type part int
	const (
		all3 part = iota
		noDeltaNet
		noPLE
	)
	restore := func(c *Checkpoint, skip part) {
		if skip == all3 {
			if err := g.Restore(c); err != nil {
				t.Fatal(err)
			}
			return
		}
		if skip != noDeltaNet {
			if err := g.dn.RestoreCarried(c.dn); err != nil {
				t.Fatal(err)
			}
		}
		if skip != noPLE {
			if err := g.ple.RestoreCarried(c.ple); err != nil {
				t.Fatal(err)
			}
		}
		if err := g.rewindPosition(c.past, c.past); err != nil {
			t.Fatal(err)
		}
	}

	var ck *Checkpoint
	run := func(skip part) [][]float32 {
		use(1)
		feed(pre, true)
		var err error
		if ck, err = g.Checkpoint(ck); err != nil {
			t.Fatal(err)
		}
		// Somewhere else first: the abandoned continuation, then the other
		// slot's conversation, then back.
		feed(other, false)
		for range steps {
			if _, _, err := g.Extend([]int32{toks[0]}); err != nil {
				t.Fatal(err)
			}
		}
		use(0)
		feed(other, true)
		use(1)
		restore(ck, skip)
		if g.Past() != prefix {
			t.Fatalf("restored to position %d, want %d", g.Past(), prefix)
		}
		_, rows := finish(toks)
		return rows
	}

	firstDiff := func(got [][]float32) string {
		for r := range want {
			for j := range want[r] {
				if got[r][j] != want[r][j] {
					return fmt.Sprintf("row %d logit %d: %g against %g", r, j, got[r][j], want[r][j])
				}
			}
		}
		return ""
	}

	if d := firstDiff(run(all3)); d != "" {
		t.Fatalf("restoring the checkpoint is not the prefill: %s", d)
	}
	t.Logf("checkpoint at %d of %d cells, %d-token detour past it and %d steps, restored in slot 1 with "+
		"slot 0 run between: %d logit rows bit-identical to prefix + tail", prefix, nKV, len(other), steps, len(want))

	for _, c := range []struct {
		name string
		skip part
	}{
		{"the deltanet state and rings", noDeltaNet},
		{"the ple ring", noPLE},
	} {
		d := firstDiff(run(c.skip))
		if d == "" {
			t.Errorf("control: a restore without %s gives the same logits, so the gate cannot see it", c.name)
			continue
		}
		t.Logf("control, without %s: moves at %s", c.name, d)
	}

	// Restore refuses a slot that no longer holds the checkpoint's tokens:
	// the cells below it are some other conversation's now.
	use(1)
	feed(other, true)
	if err := g.Restore(ck); err == nil {
		t.Error("Restore accepted a checkpoint after the slot was re-run from token zero with other tokens")
	}
}
