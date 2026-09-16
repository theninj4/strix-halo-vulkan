package llm

// L6b's gate: the five blocks in order, against llama.cpp's own logits.
//
// Every test before this one fed a block llama.cpp's value for the tensor it
// reads, so a disagreement was that block's. Here the only input is the token
// list: everything else is ours, and the tolerance has to hold error that
// accumulated through 48 layers of five block types rather than through one.
//
//	go test ./llm/ -v -run TestGraphPrefix   # four layers, ~3 s
//	go test ./llm/ -v -run TestGraphLogits   # all 48, 85 GB, ~40 s
//
// The prefix test is the one to run while the order is being changed. It
// stages four layers against `l_last-3` — which is the whole shape of the
// graph, PLE block and full-attention layer included, because the interval is
// four — and needs 7 GB instead of 85.
//
// The depth column both of them print is `reference/out/llmdepth/`, a second
// pass of the oracle over the same prompt filtered to `l_last` every eight
// layers. It exists because a single deep number is not evidence: 0.5% at the
// bottom of 48 layers is either accumulation or a bug, and only the shape of
// the curve between here and there tells them apart.

import (
	"fmt"
	"sort"
	"testing"
	"time"
)

// depthDir is the second pass of the oracle: `l_last` every eight layers of
// the same seven-token prompt, plus the head's two tensors.
//
//	/tmp/eval_dump -m $M -o reference/out/llmdepth -c 64 \
//	    -p 'The capital of France is Paris.' \
//	    -n '^l_last-(7|15|23|31|39|47)$|^result_(norm|output)$'
const depthDir = "../reference/out/llmdepth"

// graphFixture stages a graph over the 7-token trace's prompt.
func graphFixture(t *testing.T, opts GraphOpts) (*Graph, *Trace, []int32) {
	t.Helper()
	m, tr := fixtures(t)
	ids, prompt, err := tr.Tokens()
	if err != nil {
		t.Fatal(err)
	}
	dev, done := newTestDevice(t)
	t.Cleanup(done)

	opts.MaxTokens = len(ids)
	start := time.Now()
	g, err := NewGraph(dev, m, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Destroy)

	var w, a, bufs int
	for _, s := range g.Staged {
		t.Logf("  %-11s %2d layers %4d buffers %8.2f GB %8.1f MB %8s",
			s.Name, s.Layers, s.Buffers, float64(s.Weights)/1e9, float64(s.Arenas)/1e6,
			s.Elapsed.Round(time.Millisecond))
		w, a, bufs = w+s.Weights, a+s.Arenas, bufs+s.Buffers
	}
	t.Logf("%d layers, %d tokens (%q), %.2f GB of weights and %.2f GB of arenas in %d buffers, staged in %s",
		g.Layers(), len(ids), prompt, float64(w)/1e9, float64(a)/1e9, bufs,
		time.Since(start).Round(time.Millisecond))
	return g, tr, ids
}

// TestGraphPrefix runs the first four layers and checks the residual against
// `l_last-3`.
//
// Four is the whole architecture: layers 0, 1 and 2 are the gated DeltaNet,
// layer 1 has the PLE n-gram block in front of it, layer 3 is the first
// full-attention layer with the QSA indexer, and all four have a MoE block.
// So an order that is wrong anywhere is wrong here, at 9 GB and a minute
// rather than 85 GB and thirty-three seconds of staging.
func TestGraphPrefix(t *testing.T) {
	const layers = 4
	g, tr, ids := graphFixture(t, GraphOpts{Layers: layers, NoHead: true})

	// One depth per dumped `l_last`, off the same staged model. The point is
	// the *trend*: L6b's only deep comparison is at layer 47, and whether
	// 0.5% there is accumulation or a bug is a question this column answers
	// and a single number cannot.
	var last cmpResult
	for n := 1; n <= layers; n++ {
		if err := g.PrefillN(ids, n); err != nil {
			t.Fatal(err)
		}
		want, err := tr.Get(fmt.Sprintf("l_last-%d", n-1), 0)
		if err != nil {
			t.Fatal(err)
		}
		r, err := compare(g.Residual(), want.Vals)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("l_last-%d over %d tokens: %v  (%.3f%% of scale)",
			n-1, len(ids), r, 100*r.rms/r.refMax)
		last = r
	}
	// The residual is a sum of four layers of block output on top of the
	// embedding, and every one of those went through a Q8_0 matmul with an
	// fp16 accumulator (L4a-5). The per-block gates this repeats are 1e-4 to
	// 1e-3 rms on their own outputs.
	if last.rms > 5e-3 {
		t.Errorf("l_last-%d: rms %.3e over 5e-3 (%v)", layers-1, last.rms, last)
	}
}

// TestGraphLogits is L6b's gate: the whole model, and llama.cpp's logits.
//
// The comparison is the last token's row, because that is the only one the
// reference computes — `inp_out_ids` drops the rest before the final mixer,
// and `result_output` in the trace is one row of 248320 floats for a
// seven-token prompt.
//
// Three things are checked and they are not the same thing. The **depth
// column** says the disagreement is drift and not a defect. The **rms** says
// the arithmetic agrees. And the **argmax and the top ten** say the model
// agrees, which is what generation depends on and what no logit tolerance
// establishes on its own: 248320 logits within a thousandth of each other can
// still disagree about which is largest.
func TestGraphLogits(t *testing.T) {
	if testing.Short() {
		t.Skip("the whole model is 85 GB resident and 33 s of staging")
	}
	g, tr, ids := graphFixture(t, GraphOpts{})

	// 1. The drift, every eight layers. `l_last-47` is one row because
	//    `inp_out_ids` has already run by then, so the last rung compares
	//    the last token alone and the rest compare all seven.
	if dt, err := OpenTrace(depthDir); err != nil {
		t.Logf("no depth trace (%v); the curve below is not available", err)
	} else {
		wide := g.cfg.HCConfig().Wide()
		t.Logf("%-6s %-12s %10s %10s %8s", "layers", "tensor", "rms", "|ref|max", "of scale")
		for n := 8; n <= g.Layers(); n += 8 {
			name := fmt.Sprintf("l_last-%d", n-1)
			want, err := dt.Get(name, 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := g.PrefillN(ids, n); err != nil {
				t.Fatal(err)
			}
			got := g.Residual()
			if len(want.Vals) == wide {
				got = got[len(got)-wide:]
			}
			r, err := compare(got, want.Vals)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			t.Logf("%-6d %-12s %10.3e %10.3f %7.3f%%", n, name, r.rms, r.refMax, 100*r.rms/r.refMax)
		}
	}

	// 2. The forward pass the gate is about.
	start := time.Now()
	logits, norm, err := g.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("forward over %d tokens in %s", len(ids), time.Since(start).Round(time.Millisecond))

	// `result_norm` is the final mixer's output, which is what the head
	// reads. A logit disagreement is either here or in the projection, and
	// this is how the two are told apart.
	wantNorm, err := tr.Get("result_norm", 0)
	if err != nil {
		t.Fatal(err)
	}
	rn, err := compare(norm, wantNorm.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("result_norm: %v  (%.3f%% of scale)", rn, 100*rn.rms/rn.refMax)
	if rn.rms/rn.refMax > 1e-2 {
		t.Errorf("result_norm: rms %.3e is %.3f%% of scale, over 1%% (%v)", rn.rms, 100*rn.rms/rn.refMax, rn)
	}

	// 3. The logits.
	wantLog, err := tr.Get("result_output", 0)
	if err != nil {
		t.Fatal(err)
	}
	rl, err := compare(logits, wantLog.Vals)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("result_output over %d logits: %v  (%.3f%% of scale)", len(logits), rl, 100*rl.rms/rl.refMax)
	if rl.rms/rl.refMax > 5e-2 {
		t.Errorf("result_output: rms %.3e is %.3f%% of scale, over 5%% (%v)", rl.rms, 100*rl.rms/rl.refMax, rl)
	}

	// 4. The decision, which is the thing generation actually reads. The
	//    argmax has to be the reference's and the top ten have to be the
	//    same ten tokens; their *order* is reported rather than demanded,
	//    because an inversion between logits a thousandth apart is a
	//    different fact from a wrong token.
	got, want := topLogits(logits, 10), topLogits(wantLog.Vals, 10)
	t.Logf("argmax: ours %d (%.4f), llama.cpp %d (%.4f)",
		got[0], logits[got[0]], want[0], wantLog.Vals[want[0]])
	t.Logf("top 10: ours      %v", got)
	t.Logf("        llama.cpp %v", want)
	if got[0] != want[0] {
		t.Errorf("argmax %d, llama.cpp says %d", got[0], want[0])
	}
	if n := sameSet(got, want); n != len(want) {
		t.Errorf("top 10: %d of %d tokens in common", n, len(want))
	}
}

// topLogits returns the indices of the k largest values, largest first.
func topLogits(v []float32, k int) []int {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if v[idx[a]] != v[idx[b]] {
			return v[idx[a]] > v[idx[b]]
		}
		return idx[a] < idx[b]
	})
	return idx[:k]
}

// sameSet counts how many of b appear in a, order ignored.
func sameSet(a, b []int) int {
	in := make(map[int]bool, len(a))
	for _, v := range a {
		in[v] = true
	}
	n := 0
	for _, v := range b {
		if in[v] {
			n++
		}
	}
	return n
}
