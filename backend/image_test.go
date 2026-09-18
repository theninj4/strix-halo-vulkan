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
		steps, n int
		want     []int
	}{
		// The default: three frames over the checkpoint's eight steps, evenly
		// spread, the last two steps clear of the end.
		{8, 3, []int{1, 3, 5}},
		{8, 2, []int{1, 4}},
		{8, 1, []int{3}},
		{8, 0, nil},
		// More frames than there is room for. Each is clamped to the last
		// step a partial may come from and the duplicates collapse, so four
		// frames over four steps is three -- not four, and not an error.
		{4, 3, []int{0, 1, 2}},
		{3, 3, []int{0, 1}},
		{2, 3, []int{0}},
		// One step has no partials at all: its denoised estimate is the final
		// latent, so a frame there would be the finished image sent twice.
		{1, 3, nil},
		{0, 3, nil},
		// And a long schedule, to show the spacing is a fraction rather than
		// a fixed offset.
		{20, 3, []int{4, 9, 14}},
	} {
		t.Run(fmt.Sprintf("%dsteps_%dframes", c.steps, c.n), func(t *testing.T) {
			got := partialSteps(c.steps, c.n)
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
			// Nothing may come from the last step, whatever the counts.
			if c.steps > 0 {
				if _, ok := got[c.steps-1]; ok {
					t.Errorf("a partial came from the last step of %d", c.steps)
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
