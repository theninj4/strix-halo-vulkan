package backend

import (
	"slices"
	"testing"
	"time"
)

// TestPickKev is K9's scheduler on its own: costs in rows, arrival times,
// and a fit that stands in for the planner.
func TestPickKev(t *testing.T) {
	t0 := time.Unix(0, 0)
	wait := func(costs ...int) []kevWaiting {
		w := make([]kevWaiting, len(costs))
		for i, c := range costs {
			w[i] = kevWaiting{since: t0.Add(time.Duration(i) * time.Millisecond), cost: c}
		}
		return w
	}
	all := func([]int) bool { return true }
	now := t0.Add(100 * time.Millisecond)
	cases := []struct {
		name string
		w    []kevWaiting
		now  time.Time
		fits func([]int) bool
		lead int
		want []int
	}{
		{"one", wait(100), now, all, -1, []int{0}},
		{"shorts share, cheapest first", wait(120, 100, 110), now, all, -1, []int{1, 2, 0}},
		{"a long text waits for the shorts", wait(2300, 100, 100), now, all, -1, []int{1, 2}},
		{"a long text alone runs over budget", wait(2300), now, all, -1, []int{0}},
		{"the budget", wait(400, 400, 400), now, all, -1, []int{0, 1}},
		{"the batch size", wait(10, 10, 10, 10, 10, 10, 10, 10, 10), now, all, -1, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"one that does not fit is skipped, not the end",
			wait(100, 200, 300), now, func(p []int) bool { return !slices.Contains(p, 1) }, -1, []int{0, 2}},
		{"the oldest goes first once it has waited", wait(2300, 100, 100), t0.Add(time.Second), all, -1, []int{0}},
		{"an old short leads, the cheap join", wait(300, 100, 2300), t0.Add(time.Second), all, -1, []int{0, 1}},
		{"beside a stream's chunk, the budget left", wait(300, 100, 200), now, all, 512, []int{1, 2}},
		{"a stream's chunk alone", wait(600), now, all, 512, nil},
	}
	for _, c := range cases {
		got := pickKev(c.w, c.now, 8, 1024, time.Second, c.lead, c.fits)
		if !slices.Equal(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}
