package llm

// One command buffer a **pass**, and the attribution that has to survive it:
// LLM.md L7d.
//
// L7c-5 took the graph from a submit a dispatch to a submit a block — ~1130
// to ~490 — and priced the rest at 31-43 us each, which is ~15-21 ms of a
// 99 ms decode token. Nothing in the shim ever prevented going further: it
// binds each dispatch's own pipeline, descriptor set and push constants, so a
// sequence may mix blocks as freely as it mixes kernels of one block. What
// stopped it was the *measurement*. Every number this vertical has been tuned
// by comes from the host wall clock around a block's `Run`, and a block that
// no longer submits has no wall clock of its own.
//
// So the recorder moves both. Each block's `Run` appends to it instead of
// submitting when one is installed, and the submit asks the shim for a
// timestamp after **every** dispatch (vk.DispatchMultiMarked) rather than only
// at the two ends. The per-block figures that come back are then GPU time
// inside one command buffer rather than host time around many, which is the
// same thing llama.cpp's own `GGML_VK_PERF_LOGGER` reports for its graph — the
// baseline half of every comparison in LLM.md — and strictly the better
// number: it cannot be inflated by a fence wait, a Go allocation or a page
// fault on the host.

import (
	"fmt"
	"time"

	"strix-halo-vulkan/vk"
)

// The block names a dispatch can be attributed to. They are the rows of
// GraphStats, and a dispatch carries one because a command buffer does not.
const (
	ownHC   = "hc"
	ownPLE  = "ple"
	ownDN   = "dn"
	ownAttn = "attn"
	ownMoE  = "moe"
	ownHead = "head"
	ownMove = "move"
)

// maxBatch is how many dispatches go into one command buffer. It is a bound
// on the shim's query pool (SHIM_QUERY_SLOTS) and not a tuning knob: a whole
// prefill pass is ~1200 dispatches, so this is one submit for a decode token
// and two for a long prompt, against the ~490 either used to take.
const maxBatch = 1024

// MaxBatch is that bound, exported for a caller reporting how many command
// buffers a pass took.
const MaxBatch = maxBatch

// recorder collects a pass's dispatches with the block each came from.
type recorder struct {
	d     []vk.MultiDispatch
	owner []string
}

// add appends one block's sequence. A nil recorder is the direct path, which
// is what every block does when the graph is not batching and what the
// per-block benchmarks always do.
func (r *recorder) add(owner string, d []vk.MultiDispatch) bool {
	if r == nil {
		return false
	}
	r.d = append(r.d, d...)
	for range d {
		r.owner = append(r.owner, owner)
	}
	return true
}

func (r *recorder) len() int {
	if r == nil {
		return 0
	}
	return len(r.d)
}

// submit runs everything recorded, in order, and returns the GPU time each
// block's dispatches took. It chunks at maxBatch, which costs one more fence
// wait per chunk and keeps the marks.
func (r *recorder) submit() (map[string]time.Duration, time.Duration, error) {
	byOwner := make(map[string]time.Duration, 8)
	var total time.Duration
	for i := 0; i < len(r.d); i += maxBatch {
		j := minInt(i+maxBatch, len(r.d))
		el, each, err := vk.DispatchMultiMarked(r.d[i:j], true)
		if err != nil {
			return nil, 0, fmt.Errorf("llm: dispatches %d-%d: %w", i, j-1, err)
		}
		total += el
		if each == nil {
			// The shim refused the marks, which at this size it cannot; the
			// pass still ran, so report it as one block rather than losing it.
			byOwner[r.owner[i]] += el
			continue
		}
		for k, d := range each {
			byOwner[r.owner[i+k]] += d
		}
	}
	r.d, r.owner = r.d[:0], r.owner[:0]
	return byOwner, total, nil
}
