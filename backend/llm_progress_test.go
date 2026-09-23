package backend

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestProgressLines: the rates are over the interval since the last report,
// a decode rate starts at the first step rather than at the report before it,
// and requests younger than the floor are left to their completion line.
func TestProgressLines(t *testing.T) {
	t0 := time.Now()
	ctx := context.Background()
	pre := &llmJob{ctx: ctx, class: classBackground, slot: 0, prompt: make([]int32, 10000), reused: 1000,
		progress: jobProgress{enter: t0, granted: t0, prefilled: 1000, lastAt: t0, lastPrefilled: 1000}}
	dec := &llmJob{ctx: ctx, class: classInteractive, slot: 1, prompt: make([]int32, 500),
		progress: jobProgress{enter: t0, granted: t0, prefilled: 500, lastAt: t0}}
	young := &llmJob{ctx: ctx, class: classBackground, slot: 2, prompt: make([]int32, 50),
		progress: jobProgress{enter: t0.Add(9 * time.Second), granted: t0.Add(9 * time.Second), lastAt: t0.Add(9 * time.Second)}}
	wait := &llmJob{ctx: ctx, class: classBackground, slot: -1, prompt: make([]int32, 70),
		progress: jobProgress{enter: t0}}
	s := &llmSched{slots: []llmSlot{{job: pre}, {job: dec}, {job: young}}, waiting: []*llmJob{wait}}

	// 10 s in: 4000 tokens of prefill; the other decoded 50 steps, from 6 s.
	pre.progress.prefilled, pre.served = 5000, 5*time.Second
	dec.progress.decoding, dec.progress.steps = t0.Add(6*time.Second), 50
	lines := s.progressLines(t0.Add(10*time.Second), 5*time.Second)
	got := strings.Join(lines, "\n")
	t.Log("\n" + got)
	for _, want := range []string{
		"llm [background]: waiting for a slot, 10s so far, prompt 70 tokens",
		"llm [slot 0, background]: prefill 5000/10000 tokens (50%), 400 tok/s, ~13s left, 50% of the device, 10s in",
		"llm [slot 1, interactive]: generating, 51 tokens after a 500-token prompt, 12.5 tok/s",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q", want)
		}
	}
	if len(lines) != 3 {
		t.Errorf("%d lines, want 3: the 1 s-old request is below the floor", len(lines))
	}

	// The next interval's rate is over that interval alone.
	dec.progress.steps = 150
	lines = s.progressLines(t0.Add(20*time.Second), 5*time.Second)
	if got := strings.Join(lines, "\n"); !strings.Contains(got, "generating, 151 tokens after a 500-token prompt, 10.0 tok/s") {
		t.Errorf("second interval:\n%s", got)
	}
}
