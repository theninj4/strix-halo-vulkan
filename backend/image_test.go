package backend

import (
	"fmt"
	"image"
	"image/color"
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
		// Three frames over eight steps: the first from step 0, so the client
		// sees the fuzzy composition at once, the rest evenly after it.
		{0, 8, 3, []int{0, 2, 5}},
		{0, 8, 2, []int{0, 4}},
		{0, 8, 1, []int{0}},
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
		{0, 20, 3, []int{0, 6, 13}},
		// The ceiling over the default 40 steps: sixteen distinct frames,
		// 2.5 steps apart, none from the last step.
		{0, 40, 16, []int{0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37}},
		// An edit runs only the tail, and the frames spread over *that*.
		// Spread over the whole schedule instead, every one of these would
		// fall before the run started and the client would get none.
		{4, 8, 3, []int{4, 5, 6}},
		{2, 8, 3, []int{2, 4, 6}},
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
			// The first frame is the run's first step, so a client sees
			// something as soon as there is anything to see.
			if len(got) > 0 {
				if i, ok := got[c.first]; !ok || i != 0 {
					t.Errorf("frame 0 is not from step %d: %v", c.first, got)
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

// TestDrawProgress pins the preview's progress bar: red, on the bottom rows
// only, and done/steps of the width -- step 20 of 40 is half.
func TestDrawProgress(t *testing.T) {
	red := color.NRGBA{R: 255, A: 255}
	grey := color.NRGBA{R: 128, G: 128, B: 128, A: 255}
	for _, c := range []struct {
		size, done, steps, wantW, wantH int
	}{
		{1024, 20, 40, 512, 8},
		{1024, 40, 40, 1024, 8},
		{1024, 1, 40, 25, 8},
		{256, 3, 4, 192, 4}, // the height's floor
		{1024, 0, 40, 0, 0},
	} {
		t.Run(fmt.Sprintf("%d_%dof%d", c.size, c.done, c.steps), func(t *testing.T) {
			img := image.NewNRGBA(image.Rect(0, 0, c.size, c.size))
			for i := 0; i < len(img.Pix); i += 4 {
				copy(img.Pix[i:], []uint8{grey.R, grey.G, grey.B, grey.A})
			}
			drawProgress(img, c.done, c.steps)
			for y := 0; y < c.size; y++ {
				for x := 0; x < c.size; x++ {
					in := x < c.wantW && y >= c.size-c.wantH
					want := grey
					if in {
						want = red
					}
					if got := img.NRGBAAt(x, y); got != want {
						t.Fatalf("(%d,%d) is %v, want %v", x, y, got, want)
					}
				}
			}
		})
	}
}
