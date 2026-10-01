package backend

// TODO.md P20f: the served speculative loop, over the four-layer prefix
// (~20 GB: two servers, one with the draft head).
//
//	go test ./backend/ -v -run TestLLMSpeculationIsPlain

import (
	"context"
	"os"
	"strings"
	"sync"
	"testing"

	"strix-halo-vulkan/api"
)

const llmTestDraft = "../models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"

// TestLLMSpeculationIsPlain: a server with the draft head streams exactly
// what one without it streams — greedy, sampled at a fixed seed, a second
// turn that continues the first, two conversations at once, and a voice
// command restored from a checkpoint — and it speculated to do it.
//
// Sampled requests are in the comparison on purpose. A round draws each
// verified row's token with the request's sampler, in order, one draw per
// token, so with the trunk's logits unchanged (every pass is at most three
// rows, all on the row-exact decode GEMVs) the random stream is consumed
// exactly as the plain loop consumes it and the texts must agree.
//
// On four layers the draft head is rarely right, so most rounds keep one
// row; this gates the plumbing — priming, owed rows, checkpoints, the
// fallback to batched steps — not the multiplier.
func TestLLMSpeculationIsPlain(t *testing.T) {
	if _, err := os.Stat(llmTestDraft); err != nil {
		t.Skipf("no draft head at %s", llmTestDraft)
	}
	plain := newTestLLM(t, LLMOptions{Slots: 2, Reserve: 0})
	spec := newTestLLM(t, LLMOptions{Slots: 2, Reserve: 0, Draft: llmTestDraft})
	ctx := context.Background()
	run := func(l *LLM, req *api.CompletionRequest) string {
		t.Helper()
		out, err := complete(ctx, l, req)
		if err != nil {
			t.Fatal(err)
		}
		return joined(out)
	}
	sampled := func(text string, n int) *api.CompletionRequest {
		req := greedy(text, n)
		temp, seed := 0.8, int64(7)
		req.Temperature, req.Seed = &temp, &seed
		return req
	}
	turn2 := func(first string) *api.CompletionRequest {
		req := greedy("Name three rivers in Europe.", 24)
		req.Messages = append(req.Messages,
			api.Message{Role: "assistant", Content: api.MessageContent{{Type: "text", Text: first}}},
			api.Message{Role: "user", Content: api.MessageContent{{Type: "text", Text: "And three in Asia?"}}})
		return req
	}
	var sys strings.Builder
	sys.WriteString("You are a voice assistant for a smart home. The devices:\n")
	for i := range 30 {
		sys.WriteString("- light.room_" + string(rune('a'+i%26)) + ": off\n")
	}

	// Each server's slots once over fresh arenas first (TestLLMConcurrentIsSolo).
	for _, l := range []*LLM{plain, spec} {
		run(l, greedy("Hello.", 4))
		run(l, greedy("Hi.", 4))
	}

	type check struct {
		name string
		do   func(l *LLM) []string
	}
	checks := []check{
		{"greedy", func(l *LLM) []string { return []string{run(l, greedy("Write a haiku about the sea.", 32))} }},
		{"sampled", func(l *LLM) []string {
			return []string{run(l, sampled("Tell me about owls, at length.", 32))}
		}},
		{"two turns", func(l *LLM) []string {
			first := run(l, greedy("Name three rivers in Europe.", 24))
			return []string{first, run(l, turn2(first))}
		}},
		{"concurrent, then alone", func(l *LLM) []string {
			out := make([]string, 3)
			var wg sync.WaitGroup
			for i, p := range []string{"What is 17 times 23?", "Describe a cat in detail."} {
				wg.Add(1)
				go func() {
					defer wg.Done()
					s, err := complete(ctx, l, greedy(p, 24))
					if err != nil {
						t.Error(err)
					}
					out[i] = joined(s)
				}()
			}
			wg.Wait()
			// A second turn on whichever slot the cat went to: its draft
			// owes the batched rows, and must have caught up on them.
			req := greedy("Describe a cat in detail.", 24)
			req.Messages = append(req.Messages,
				api.Message{Role: "assistant", Content: api.MessageContent{{Type: "text", Text: out[1]}}},
				api.Message{Role: "user", Content: api.MessageContent{{Type: "text", Text: "And a dog?"}}})
			out[2] = run(l, req)
			return out
		}},
		{"checkpoint restored", func(l *LLM) []string {
			a := run(l, voiceReq(sys.String(), "Turn on the light in room c.", 16))
			b := run(l, voiceReq(sys.String(), "Is the light in room d on?", 16))
			return []string{a, b}
		}},
	}
	for _, c := range checks {
		t.Run(c.name, func(t *testing.T) {
			r0, b0 := spec.sched.rounds, spec.sched.batches
			want := c.do(plain)
			got := c.do(spec)
			rounds, batches := spec.sched.rounds-r0, spec.sched.batches-b0
			for i := range want {
				if want[i] == "" {
					t.Errorf("text %d is empty, so it compares nothing", i)
				}
				if got[i] != want[i] {
					t.Errorf("text %d with the draft head:\n  %q\nwithout:\n  %q", i, got[i], want[i])
				}
			}
			if rounds == 0 {
				t.Error("no speculative round ran")
			}
			t.Logf("%d speculative rounds, %d batched passes; %q", rounds, batches, got[len(got)-1])
		})
	}
	if spec.sched.restores == 0 {
		t.Error("no checkpoint was restored")
	}
}
