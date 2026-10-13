package audiobox

import (
	"os"
	"sort"
	"testing"
	"time"

	"strix-halo-vulkan/vk"
)

// TestProfile times every dispatch of a full batch of real windows, summed
// by kind (AUDIOBOX_PROFILE=1).
func TestProfile(t *testing.T) {
	if os.Getenv("AUDIOBOX_PROFILE") == "" {
		t.Skip("AUDIOBOX_PROFILE=1")
	}
	loadManifest(t)
	g := load(t).g
	wins, valid, _ := windows(readF32(t, "lofi_wav16"))
	wins, valid = wins[:g.maxWin], valid[:g.maxWin]
	if _, _, err := g.run(wins, valid, layers); err != nil {
		t.Fatal(err)
	}
	gr := &graph{g: g}
	for b := range wins {
		gr.extract(b)
	}
	gr.encode(valid, layers)
	by := map[string]time.Duration{}
	var total time.Duration
	for i := range gr.d {
		d, err := vk.DispatchMultiTimed(gr.d[i:i+1], 1, 1, true)
		if err != nil {
			t.Fatal(err)
		}
		by[gr.kinds[i]] += d
		total += d
	}
	var ks []string
	for k := range by {
		ks = append(ks, k)
	}
	sort.Slice(ks, func(i, j int) bool { return by[ks[i]] > by[ks[j]] })
	t.Logf("%d windows, %d dispatches, %v", len(wins), len(gr.d), total)
	for _, k := range ks {
		t.Logf("  %-18s %8.2f ms  %4.1f%%", k, by[k].Seconds()*1e3, 100*by[k].Seconds()/total.Seconds())
	}
}
