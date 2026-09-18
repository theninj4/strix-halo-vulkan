package backend

import (
	"fmt"
	"sort"
	"testing"
)

// TestPartialSteps pins where in-progress frames come from.
//
// It is host arithmetic with no device in it, which is why it is worth a test
// of its own: the rule has three edges (fewer steps than frames, a single-step
// schedule, and the last step) and none of them would show up in an image.
func TestPartialSteps(t *testing.T) {
	for _, c := range []struct {
		first, steps, n int
		want            []int
	}{
		// The default: three frames over the checkpoint's eight steps, evenly
		// spread, the last two steps clear of the end.
		{0, 8, 3, []int{1, 3, 5}},
		{0, 8, 2, []int{1, 4}},
		{0, 8, 1, []int{3}},
		{0, 8, 0, nil},
		// More frames than there is room for. Each is clamped to the last
		// step a partial may come from and the duplicates collapse, so four
		// frames over four steps is three -- not four, and not an error.
		{0, 4, 3, []int{0, 1, 2}},
		{0, 3, 3, []int{0, 1}},
		{0, 2, 3, []int{0}},
		// One step has no partials at all: its denoised estimate is the final
		// latent, so a frame there would be the finished image sent twice.
		{0, 1, 3, nil},
		{0, 0, 3, nil},
		// And a long schedule, to show the spacing is a fraction rather than
		// a fixed offset.
		{0, 20, 3, []int{4, 9, 14}},
		// An edit runs only the tail, and the frames spread over *that*.
		// Spread over the whole schedule instead, every one of these would
		// fall before the run started and the client would get none.
		{4, 8, 3, []int{4, 5, 6}},
		{2, 8, 3, []int{2, 4, 5}},
		{6, 8, 3, []int{6}},
		{7, 8, 3, nil},
	} {
		t.Run(fmt.Sprintf("from%d_%dsteps_%dframes", c.first, c.steps, c.n), func(t *testing.T) {
			got := partialSteps(c.first, c.steps, c.n)
			if len(got) != len(c.want) {
				t.Fatalf("%v, want %v", keys(got), c.want)
			}
			keyList := keys(got)
			for i, k := range keyList {
				if k != c.want[i] {
					t.Fatalf("%v, want %v", keyList, c.want)
				}
				// The frame indices are 0..n-1 in step order, because that is
				// what OpenAI's partial_image_index means.
				if got[k] != i {
					t.Errorf("step %d is frame %d, want %d", k, got[k], i)
				}
			}
			// Nothing may come from the last step, whatever the counts, and
			// nothing from before the run began.
			if c.steps > 0 {
				if _, ok := got[c.steps-1]; ok {
					t.Errorf("a partial came from the last step of %d", c.steps)
				}
			}
			for _, k := range keyList {
				if k < c.first {
					t.Errorf("a partial came from step %d, before the run's first (%d)", k, c.first)
				}
			}
		})
	}
}

func keys(m map[int]int) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}
