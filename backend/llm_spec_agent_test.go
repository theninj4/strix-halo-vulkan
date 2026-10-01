package backend

// The served loop over an agent's turns, over the four-layer prefix (~20 GB:
// two servers, one with the draft head).
//
//	go test ./backend/ -v -run TestLLMSpeculationAgentTurns

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"strix-halo-vulkan/api"
)

// TestLLMSpeculationAgentTurns: an agent's conversation, each turn the last
// one's messages, its reply and a tool result, streams with the draft head
// what it streams without — and the draft is primed through every turn.
//
// A turn whose reply comes back other than the slot generated it diverges
// before the end of what the slot holds, so it restores the checkpoint at
// the last user message and prefills from there; its first chunk ends at the
// new mark and re-takes the checkpoint into the one it restored. That moved
// ck.Past() under the slot's bookkeeping, which then held the chunk's tokens
// twice: the next chunk was refused with "priming the draft at N; its cache
// holds M", and without a draft head the slot's held sequence was silently
// not the device's. PreemptChunk cuts the prefill so a chunk follows it.
func TestLLMSpeculationAgentTurns(t *testing.T) {
	if _, err := os.Stat(llmTestDraft); err != nil {
		t.Skipf("no draft head at %s", llmTestDraft)
	}
	opt := LLMOptions{Slots: 2, Reserve: 0, PreemptChunk: 64}
	plain := newTestLLM(t, opt)
	opt.Draft = llmTestDraft
	spec := newTestLLM(t, opt)
	ctx := context.Background()
	text := func(s string) api.MessageContent { return api.MessageContent{{Type: "text", Text: s}} }
	var sys strings.Builder
	sys.WriteString("You are an agent. The tools:\n")
	for i := range 40 {
		fmt.Fprintf(&sys, "- tool_%d: does thing %d\n", i, i)
	}
	// turns runs the conversation on l; every other reply goes back edited.
	turns := func(l *LLM) ([]string, error) {
		req := greedy("Tidy the workspace.", 24)
		req.Messages = append([]api.Message{{Role: "system", Content: text(sys.String())}}, req.Messages...)
		var out []string
		for turn := range 4 {
			s, err := complete(ctx, l, req)
			if err != nil {
				return out, fmt.Errorf("turn %d: %w", turn, err)
			}
			reply := joined(s)
			out = append(out, reply)
			if turn%2 == 0 {
				reply = strings.TrimSpace(reply) + " (edited)"
			}
			req.Messages = append(req.Messages, api.Message{Role: "assistant", Content: text(reply)},
				api.Message{Role: "user", Content: text(fmt.Sprintf("Result %d: %s", turn, strings.Repeat("ok ", 20+7*turn)))})
		}
		return out, nil
	}
	want, err := turns(plain)
	if err != nil {
		t.Fatalf("without the draft head: %v", err)
	}
	r0, k0 := spec.sched.rounds, spec.sched.restores
	got, err := turns(spec)
	if err != nil {
		t.Fatalf("with the draft head: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("turn %d with the draft head:\n  %q\nwithout:\n  %q", i, got[i], want[i])
		}
	}
	rounds, restores := spec.sched.rounds-r0, spec.sched.restores-k0
	if rounds == 0 {
		t.Error("no speculative round ran")
	}
	if restores == 0 {
		t.Error("no checkpoint was restored")
	}
	t.Logf("%d speculative rounds, %d restores", rounds, restores)
}
