package backend

// CONCURRENCY.md C2: the scheduler, over the four-layer prefix (~8 GB, a few
// seconds to stage).
//
//	go test ./backend/ -v -run 'TestLLM(Concurrent|Interactive)'

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/api"
)

const llmTestModel = "../models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

func newTestLLM(t *testing.T, opt LLMOptions) *LLM {
	t.Helper()
	if _, err := os.Stat(llmTestModel); err != nil {
		t.Skipf("no checkpoint at %s", llmTestModel)
	}
	dev, err := OpenDevice("llm-sched-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	t.Cleanup(dev.Close)
	opt.Model, opt.Device, opt.Layers = llmTestModel, dev, 4
	opt.Context, opt.Batch = 1024, 256
	l, err := NewLLM(opt)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(l.Close)
	return l
}

// greedy is a request that decodes deterministically for n tokens.
func greedy(text string, n int) *api.CompletionRequest {
	zero, seed := 0.0, int64(1)
	return &api.CompletionRequest{
		Messages:    []api.Message{{Role: "user", Content: api.MessageContent{{Type: "text", Text: text}}}},
		MaxTokens:   n,
		Temperature: &zero,
		Seed:        &seed,
	}
}

// stamp is one delta and when it arrived.
type stamp struct {
	at   time.Time
	text string
}

// complete runs one request and returns everything it streamed, stamped.
func complete(ctx context.Context, l *LLM, req *api.CompletionRequest) ([]stamp, error) {
	var out []stamp
	_, err := l.Complete(ctx, req, func(d api.Delta) error {
		out = append(out, stamp{time.Now(), d.ReasoningContent + d.Content})
		return nil
	})
	return out, err
}

func joined(s []stamp) string {
	var b strings.Builder
	for _, x := range s {
		b.WriteString(x.text)
	}
	return b.String()
}

// TestLLMConcurrentIsSolo: three greedy conversations run at once, through
// three slots and one graph, stream exactly the text each streams alone. Their
// decode steps share passes (C5), which the test checks happened. Any
// state two slots shared, any unit run on the wrong slot, any held prefix
// credited to the wrong conversation, changes a token.
func TestLLMConcurrentIsSolo(t *testing.T) {
	const n = 24
	l := newTestLLM(t, LLMOptions{Slots: 3, Reserve: 0, PreemptChunk: 64})
	prompts := []string{
		"Name three rivers in Europe.",
		"Write a haiku about the sea, and then explain it in a long paragraph that goes on and on.",
		"What is 17 times 23?",
	}
	ctx := context.Background()

	concurrent := func() []string {
		out := make([]string, len(prompts))
		var wg sync.WaitGroup
		for i, p := range prompts {
			wg.Add(1)
			go func() {
				defer wg.Done()
				s, err := complete(ctx, l, greedy(p, n))
				if err != nil {
					t.Error(err)
				}
				out[i] = joined(s)
			}()
		}
		wg.Wait()
		return out
	}

	// A first run over fresh arenas is not bit-repeatable against later ones
	// (llm's TestPrerecordedDecodeIsTheRecordedDecode), so every slot is
	// used once before anything is compared.
	concurrent()
	var solo []string
	for _, p := range prompts {
		s, err := complete(ctx, l, greedy(p, n))
		if err != nil {
			t.Fatal(err)
		}
		solo = append(solo, joined(s))
	}
	b0, r0 := l.sched.batches, l.sched.batchRows
	got := concurrent()
	b, r := l.sched.batches-b0, l.sched.batchRows-r0
	if b == 0 {
		t.Error("no decode step shared a pass, so the comparison says nothing about batching (C5)")
	} else {
		t.Logf("%d batched decode passes, %.2f rows each", b, float64(r)/float64(b))
	}
	for i := range prompts {
		if got[i] != solo[i] {
			t.Errorf("conversation %d concurrently:\n  %q\nalone:\n  %q", i, got[i], solo[i])
		}
		if solo[i] == "" {
			t.Errorf("conversation %d streamed nothing, so it compares nothing", i)
		}
	}
}

// TestLLMInteractiveGoesFirst: two background conversations are decoding when
// an interactive one arrives, and from its first token to its last neither
// streams more than the one unit that may already have been on the device.
//
// The control is the same arrival without the service tier: then the three
// share the device, and the background ones stream throughout. If they did
// not, the gate's window would be too short to see anything.
func TestLLMInteractiveGoesFirst(t *testing.T) {
	l := newTestLLM(t, LLMOptions{Slots: 3, Reserve: 0, PreemptChunk: 64})
	for _, tier := range []string{"priority", ""} {
		during := interactiveScenario(t, l, tier)
		for i, n := range during {
			switch {
			case tier == "priority" && n > 1:
				t.Errorf("background %d streamed %d deltas while an interactive request was live; at most the "+
					"one unit already on the device may finish", i, n)
			case tier == "" && n <= 1:
				t.Errorf("control: background %d streamed %d deltas beside a request of its own class, so the "+
					"gate's window is too short to see a priority", i, n)
			}
		}
	}
}

// interactiveScenario starts two long background requests, waits for both to
// stream, then runs a short one with the given service tier, and returns how
// many deltas each background request streamed while it did.
func interactiveScenario(t *testing.T, l *LLM, tier string) [2]int {
	ctx := context.Background()
	var mu sync.Mutex
	var bg [2][]stamp
	var wg sync.WaitGroup
	started := make(chan struct{}, 2)
	for i := range bg {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := greedy(fmt.Sprintf("Background task %d: write a long story about a lighthouse.", i), 160)
			once := false
			_, err := l.Complete(ctx, req, func(d api.Delta) error {
				mu.Lock()
				bg[i] = append(bg[i], stamp{time.Now(), d.ReasoningContent + d.Content})
				mu.Unlock()
				if !once {
					once = true
					started <- struct{}{}
				}
				return nil
			})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	<-started
	<-started

	req := greedy("Turn on the kitchen lights.", 16)
	req.ServiceTier = tier
	begin := time.Now()
	fg, err := complete(ctx, l, req)
	end := time.Now()
	if err != nil {
		t.Fatal(err)
	}
	if len(fg) == 0 {
		t.Fatal("the short request streamed nothing")
	}
	first := fg[0].at
	wg.Wait()

	var during [2]int
	for i := range bg {
		after := 0
		for _, s := range bg[i] {
			switch {
			case s.at.After(first) && s.at.Before(end):
				during[i]++
			case !s.at.Before(end):
				after++
			}
		}
		t.Logf("tier %q: background %d streamed %d deltas while the short request did, %d after it",
			tier, i, during[i], after)
		if after == 0 {
			t.Errorf("tier %q: background %d had finished before the short request did, so this run shows nothing",
				tier, i)
		}
	}
	t.Logf("tier %q: short request's first delta after %v, done after %v", tier,
		first.Sub(begin).Round(time.Millisecond), end.Sub(begin).Round(time.Millisecond))
	return during
}

// voiceReq is a Home-Assistant-shaped request: a long system prompt that
// every command shares, and a short utterance.
func voiceReq(system, utterance string, n int) *api.CompletionRequest {
	req := greedy(utterance, n)
	req.Messages = append([]api.Message{{Role: "system",
		Content: api.MessageContent{{Type: "text", Text: system}}}}, req.Messages...)
	return req
}

// TestLLMCheckpointIsTheSystemPrompt is C4's gate through the backend. A voice
// command whose system prompt the slot has checkpointed restores the
// checkpoint and prefills only its own turn, and streams **exactly** the text
// it streams when the whole prompt is prefilled fresh.
//
// Both runs cut the prefill at the same boundary (the checkpoint's), so the
// comparison is between a restored state and a state computed in place, with
// the same chunks after it. The restore counter is checked too, because a
// matching answer from a run that never restored would pass the comparison
// and say nothing.
func TestLLMCheckpointIsTheSystemPrompt(t *testing.T) {
	l := newTestLLM(t, LLMOptions{Slots: 1})
	var sys, other strings.Builder
	sys.WriteString("You are a voice assistant for a smart home. The devices:\n")
	other.WriteString("You are a librarian. The shelves:\n")
	for i := range 40 {
		fmt.Fprintf(&sys, "- light.room_%d: off, brightness 0\n", i)
		fmt.Fprintf(&other, "- shelf %d: history, volumes %d to %d\n", i, 10*i, 10*i+9)
	}
	ctx := context.Background()
	run := func(req *api.CompletionRequest) string {
		out, err := complete(ctx, l, req)
		if err != nil {
			t.Fatal(err)
		}
		return joined(out)
	}
	ask := voiceReq(sys.String(), "Turn on the light in room 7.", 24)

	// Fresh: the slot holds a different system prompt, so nothing is reused,
	// and the run checkpoints this one at the boundary it cuts its prefill at.
	run(voiceReq(other.String(), "Where is volume 12?", 8))
	want := run(ask)
	if n := l.sched.restores; n != 0 {
		t.Fatalf("%d restores before any checkpoint of this system prompt", n)
	}
	// A different command restores it and runs on past it, so the next
	// restore has a detour to undo; then the same command again.
	run(voiceReq(sys.String(), "Is the light in room 3 on?", 8))
	got := run(ask)
	if n := l.sched.restores; n != 2 {
		t.Fatalf("%d restores; the second and third commands should each have restored the system prompt", n)
	}
	if got != want {
		t.Fatalf("restored from the checkpoint:\n%q\nprefilled fresh:\n%q", got, want)
	}
	t.Logf("restored from the checkpoint and streamed the fresh prefill's %d bytes exactly: %q", len(got), got)
}
