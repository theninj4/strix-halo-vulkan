package pipeline

import (
	"log"
	"os"
	"runtime"
	"strings"
	"time"
)

// memLog, when H3_MEMLOG is set, logs the Go heap and the process's
// resident anonymous memory every 250 ms until stop is called: a diagnostic
// for where a request's host memory goes (VIDEO.md M11a).
func memLog(tag string) (stop func()) {
	if os.Getenv("H3_MEMLOG") == "" {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		t0 := time.Now()
		tick := time.NewTicker(250 * time.Millisecond)
		defer tick.Stop()
		for {
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			log.Printf("memlog %s +%.2fs heapAlloc %.2f GB heapInuse %.2f heapIdle %.2f heapReleased %.2f next %.2f numGC %d rssAnon %s",
				tag, time.Since(t0).Seconds(), gb(m.HeapAlloc), gb(m.HeapInuse), gb(m.HeapIdle), gb(m.HeapReleased), gb(m.NextGC), m.NumGC, rssAnon())
			select {
			case <-done:
				return
			case <-tick.C:
			}
		}
	}()
	return func() { close(done) }
}

func gb(n uint64) float64 { return float64(n) / 1e9 }

func rssAnon() string {
	b, _ := os.ReadFile("/proc/self/status")
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "RssAnon:") {
			return strings.TrimSpace(strings.TrimPrefix(l, "RssAnon:"))
		}
	}
	return "?"
}
