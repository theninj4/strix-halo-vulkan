package main

// P5a — the draft head as a passive observer (LLM.md P5,
// research/p5-mtp-rollback.md).
//
//	go run ./cmd/llm -mtp -mtp-n 256 -mtp-depth 4
//	go run ./cmd/llm -mtp -mtp-free                      # the model's own text
//	go run ./cmd/llm -mtp -mtp-wire res,mixed,zero       # screen the wirings
//
// **The number this exists for is the acceptance rate, and it needs none of
// P5's machinery.** A speculative loop wants to know how often the draft
// head's guess equals the token the trunk would have emitted anyway — and
// that can be watched from the side of an ordinary one-token-at-a-time
// decode, with no rollback, no verification pass and no new kernel:
//
//	at step i the trunk consumes x_i and its logits name y_{i+1} = argmax
//	the draft is handed (h_i, x_{i+1}) at position i+1 and names x̂_{i+2}
//	at step i+1 the trunk names y_{i+2}, and the two are compared
//
// Two modes, because they measure different things and the honest answer is
// both. **teacher-forced** (the default) advances on the corpus's own tokens,
// so the context stays on-distribution and the comparison is still against
// the trunk's argmax — this is the conservative reading. **free-running**
// advances on the trunk's argmax, which is exactly what a speculative loop
// would see, and is the optimistic one because a model's own greedy output is
// more predictable than real text.
//
// Free-running is **primed** (`-mtp-prime`, default 64) rather than started
// from one token, and the generated text is printed. Both are there for the
// same reason: greedy decoding off a single token degenerates into repetition
// within a few dozen steps, and a repetitive trunk is trivially predictable —
// so an unprimed free-running number would measure the degeneration and not
// the draft head. Reading the text is the check that it did not happen
// anyway.
//
// **The trunk is walked once and every wiring is replayed against it.** There
// is no reference implementation of this architecture's MTP block anywhere
// (llm/mtp.go says where it is not), so the reading of `nextn` is a
// hypothesis and the honest thing is to measure the candidates against each
// other rather than to pick one. Restaging the trunk per arm would be a
// minute each and would compare arms across different machine states;
// instead the walk records `(h_i, result_norm_i, token, argmax_i)` for every
// step — 13 MB for 256 of them — and each arm replays the draft over those
// rows with a **fresh draft cache**, so no arm can contaminate another's KV
// history.
//
// The arms, and what each is a hypothesis about:
//
//	res         the trunk's 10240-wide hyper-connection residual, per stream
//	res-flip    the same, with the concatenation the other way round
//	mixed       the trunk's collapsed `result_norm`, broadcast into 4 streams
//	mixed-flip  the same, flipped
//	zero        a zeroed hidden state — **the negative control**
//
// `zero` is the one that makes the others mean anything: a head that scores
// well on it is predicting from the token embedding alone, and no acceptance
// rate from any wiring arm would then be evidence about the wiring.

import (
	"fmt"
	"os"
	"strings"
	"time"

	"strix-halo-vulkan/llm"
)

type mtpOpts struct {
	model, draft string
	file, prompt string
	n, ctx       int
	layers       int
	depth        int
	free         bool
	prime        int
	wires        string
	csv          string
}

// mtpArm is one reading of the `nextn` block, resolved from its name.
type mtpArm struct {
	name      string
	wiring    llm.Wiring
	collapsed bool // hand the block `result_norm` broadcast, not the wide residual
	zero      bool // the negative control
}

func parseArms(s string) ([]mtpArm, error) {
	var out []mtpArm
	for _, name := range strings.Split(s, ",") {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		base, flip := strings.CutSuffix(name, "-flip")
		a := mtpArm{name: name, wiring: llm.Wiring{Name: name, Flip: flip}}
		switch base {
		case "res":
		case "mixed":
			a.collapsed = true
		case "zero":
			a.zero = true
		default:
			return nil, fmt.Errorf("unknown wiring %q: want res, res-flip, mixed, mixed-flip or zero", name)
		}
		out = append(out, a)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no wirings named")
	}
	return out, nil
}

// trunkStep is what one pass of the trunk leaves behind for the draft to be
// replayed against.
type trunkStep struct {
	res    []float32 // the wide hyper-connection residual, [hc*nEmbd]
	mixed  []float32 // `result_norm`, [nEmbd]
	fed    int32     // the token the sequence advanced on after this step
	greedy int32     // the trunk's own argmax from this position
}

// mtpObserve stages the trunk and the draft head, walks a sequence one token
// at a time, then replays each wiring over the recorded walk.
func mtpObserve(o mtpOpts) error {
	arms, err := parseArms(o.wires)
	if err != nil {
		return err
	}
	m, err := llm.Open(o.model)
	if err != nil {
		return err
	}
	defer m.Close()
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		return err
	}

	// The text. A corpus file keeps the context on-distribution; -prompt is
	// the short-input arm, and in free-running mode it is all that is read.
	var ids []int32
	if o.prompt != "" {
		if ids, err = tok.Encode(o.prompt); err != nil {
			return err
		}
	} else {
		buf, err := os.ReadFile(o.file)
		if err != nil {
			return err
		}
		if ids, err = tok.Encode(string(buf)); err != nil {
			return err
		}
	}
	steps := o.n + o.depth + 1
	if len(ids) < steps+1 && !o.free {
		return fmt.Errorf("the text tokenizes to %d tokens, this run needs %d", len(ids), steps+1)
	}

	draft, err := llm.Open(o.draft)
	if err != nil {
		return fmt.Errorf("draft head: %w", err)
	}
	defer draft.Close()

	dev, done, err := openDevice()
	if err != nil {
		return err
	}
	defer done()

	ctx := o.ctx
	if want := steps + 4; ctx < want {
		ctx = want
	}
	fmt.Printf("%s\n  + %s\n", o.model, o.draft)
	fmt.Printf("%d trunk layers, draft blk.%d, %d cells of context, %d rounds, depth %d, %s\n",
		m.Config.NLayer, llm.MTPLayer(m.Config), ctx, o.n, o.depth,
		map[bool]string{true: "free-running", false: "teacher-forced"}[o.free])

	start := time.Now()
	g, err := llm.NewGraph(dev, m, llm.GraphOpts{MaxTokens: 1, NKV: ctx, Layers: o.layers})
	if err != nil {
		return err
	}
	defer g.Destroy()
	fmt.Printf("trunk staged in %v\n", time.Since(start).Round(time.Millisecond))

	start = time.Now()
	d, err := llm.NewMTPHead(dev, draft, m.Config, g.Head(), 1, ctx)
	if err != nil {
		return err
	}
	defer d.Destroy()
	fmt.Printf("draft staged in %v: %.2f GB of weights, %.2f GB of arenas (the lm head is the trunk's)\n",
		time.Since(start).Round(time.Millisecond),
		float64(d.WeightBytes())/1e9, float64(d.ActivationBytes())/1e9)

	// 1. One walk of the trunk, recorded.
	if err := g.Reset(); err != nil {
		return err
	}
	walk := make([]trunkStep, 0, steps)
	begin := time.Now()
	cur := ids[0]
	for i := 0; i < steps; i++ {
		logits, norm, err := g.Extend([]int32{cur})
		if err != nil {
			return fmt.Errorf("trunk step %d: %w", i, err)
		}
		next := llm.Argmax(logits)
		adv := next
		if (!o.free || i+1 < o.prime) && i+1 < len(ids) {
			adv = ids[i+1]
		}
		// Both are slices into mapped arenas the next pass overwrites.
		walk = append(walk, trunkStep{
			res:    append([]float32(nil), g.Residual()...),
			mixed:  append([]float32(nil), norm...),
			fed:    adv,
			greedy: next,
		})
		cur = adv
		if (i+1)%32 == 0 {
			fmt.Printf("  trunk %d/%d\r", i+1, steps)
		}
	}
	trunkWall := time.Since(begin)
	fmt.Printf("%d trunk steps in %v (%.1f ms each)\n",
		steps, trunkWall.Round(time.Millisecond),
		float64(trunkWall.Microseconds())/1000/float64(steps))
	if o.free {
		// Printed so the degeneration check is a reading rather than an
		// assumption: a free-running acceptance rate is only about the draft
		// head if the trunk was not looping.
		fed := make([]int32, len(walk))
		for i, w := range walk {
			fed[i] = w.fed
		}
		if txt, err := tok.Decode(fed[o.prime:]); err == nil {
			fmt.Printf("\n--- free-running, %d primed then %d greedy ---\n%s\n---\n",
				o.prime, len(fed)-o.prime, txt)
		}
	}

	// 2. Each wiring, replayed over the same walk with a fresh draft cache.
	zero := make([]float32, d.Wide())
	rows := [][]string{{"wiring", "k", "tried", "hit", "a_k", "chain_tried", "chain_hit", "chain_a_k", "e_tokens"}}
	for _, a := range arms {
		d.Reset()
		hit := make([]int, o.depth)
		tried := make([]int, o.depth)
		chainHit := make([]int, o.depth)
		chainTried := make([]int, o.depth)

		begin = time.Now()
		seed0, gpu0, steps0 := d.SeedTime, d.GPUTime, d.Steps
		for i := 0; i < o.n; i++ {
			var h []float32
			switch {
			case a.zero:
				h = zero
			case a.collapsed:
				if h, err = d.Broadcast(walk[i].mixed); err != nil {
					return err
				}
			default:
				h = walk[i].res
			}
			// The chain names positions i+2, i+3, … and the trunk's answer
			// for position j is its argmax from position j-1.
			guess, err := d.Chain(h, m, walk[i].fed, i+1, o.depth, a.wiring, a.collapsed)
			if err != nil {
				return fmt.Errorf("%s: draft round %d: %w", a.name, i, err)
			}
			alive := true
			for k, w := range guess {
				j := i + 1 + k
				if j >= len(walk) {
					break
				}
				good := w == walk[j].greedy
				tried[k]++
				if good {
					hit[k]++
				}
				if alive {
					chainTried[k]++
					if good {
						chainHit[k]++
					} else {
						alive = false
					}
				}
			}
			if (i+1)%64 == 0 {
				fmt.Printf("  %s %d/%d\r", a.name, i+1, o.n)
			}
		}
		wall := time.Since(begin)

		nstep := max(d.Steps-steps0, 1)
		fmt.Printf("\n%-11s %d rounds x depth %d in %v — %.2f ms a draft step: "+
			"%.2f host `nextn`, %.2f the layer's %d submits\n",
			a.name, o.n, o.depth, wall.Round(time.Millisecond),
			float64(wall.Microseconds())/1000/float64(nstep),
			float64((d.SeedTime-seed0).Microseconds())/1000/float64(nstep),
			float64((d.GPUTime-gpu0).Microseconds())/1000/float64(nstep), 13)
		fmt.Printf("  k   marginal a_k          chained a_k         E[tokens] at depth k\n")
		exp := 1.0
		for k := 0; k < o.depth; k++ {
			var ak, cak, surv float64
			if tried[k] > 0 {
				ak = float64(hit[k]) / float64(tried[k])
			}
			if chainTried[k] > 0 {
				cak = float64(chainHit[k]) / float64(chainTried[k])
			}
			if chainTried[0] > 0 {
				// The measured survival fraction, not a power of a_1: a
				// round of depth k+1 emits one token plus however many of
				// its drafts were right from the front.
				surv = float64(chainHit[k]) / float64(chainTried[0])
			}
			exp += surv
			fmt.Printf("%3d   %5d/%5d %6.1f%%    %5d/%5d %6.1f%%    %6.3f\n",
				k+1, hit[k], tried[k], 100*ak, chainHit[k], chainTried[k], 100*cak, exp)
			rows = append(rows, []string{
				a.name, fmt.Sprint(k + 1), fmt.Sprint(tried[k]), fmt.Sprint(hit[k]), fmt.Sprintf("%.4f", ak),
				fmt.Sprint(chainTried[k]), fmt.Sprint(chainHit[k]), fmt.Sprintf("%.4f", cak),
				fmt.Sprintf("%.4f", exp),
			})
		}
	}

	if o.csv != "" {
		if err := writePPL(o.csv, rows); err != nil {
			return err
		}
		fmt.Printf("\nwrote %s\n", o.csv)
	}
	return nil
}
