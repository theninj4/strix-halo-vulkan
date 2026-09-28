// Command kevload drives a served Kev (`cmd/serve -kev`) with an open-loop
// mix of short and long System One requests and reports the latency each
// class sees (CLASSIFICATION.md K9). Every request has a text of its own, so
// the prefix cache never answers one.
//
//	go run ./cmd/kevload -url http://127.0.0.1:8099 -rate 15 -long-every 2s -for 20s
//	go run ./cmd/kevload -url http://127.0.0.1:8099 -burst 16 -burst-long 1
//
// A short request is the README ticket (3 questions, ~100 packed tokens). A
// long one is the long fixture's report seven times over, with its questions
// and two of the ticket's (~2,300 tokens), as in TestGPUCacheLatency.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type fixture struct {
	Name    string         `json:"name"`
	Request map[string]any `json:"request"`
}

type result struct {
	long bool
	d    time.Duration
	err  error
}

func main() {
	log.SetFlags(0)
	url := flag.String("url", "http://127.0.0.1:8099", "the server")
	fixtures := flag.String("fixtures", "reference/kev_fixtures.json", "Kev's fixtures")
	rate := flag.Float64("rate", 0, "short requests a second, Poisson arrivals (open loop)")
	longEvery := flag.Duration("long-every", 0, "one long request this often, alongside the shorts (0: none)")
	dur := flag.Duration("for", 20*time.Second, "how long arrivals go on")
	burst := flag.Int("burst", 0, "instead of arrivals: this many short requests at once")
	burstLong := flag.Int("burst-long", 0, "and this many long ones in the same burst, sent first")
	seed := flag.Int64("seed", 1, "arrival seed")
	warm := flag.Int("warm", 3, "unmeasured short requests first, one at a time")
	flag.Parse()

	raw, err := os.ReadFile(*fixtures)
	if err != nil {
		log.Fatal(err)
	}
	var fx []fixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		log.Fatal(err)
	}
	short, long := fx[0].Request, fx[2].Request
	n := 0
	body := func(isLong bool) []byte {
		n++
		var req map[string]any
		if isLong {
			qs := map[string]any{}
			for k, v := range long["questions"].(map[string]any) {
				qs[k] = v
			}
			i := 0
			for k, v := range short["questions"].(map[string]any) {
				if i == 2 {
					break
				}
				qs["t_"+k] = v
				i++
			}
			text := strings.Repeat(long["state"].(string)+"\n\n", 7)
			req = map[string]any{"model": "kev-latest", "state": fmt.Sprintf("Report %d. %s", n, text), "questions": qs}
		} else {
			req = map[string]any{"model": "kev-latest", "state": fmt.Sprintf("Ticket %d. %s", n, short["state"]), "questions": short["questions"]}
		}
		b, _ := json.Marshal(req)
		return b
	}
	do := func(b []byte) error {
		resp, err := http.Post(*url+"/v1/systemone", "application/json", bytes.NewReader(b))
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		out, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			return fmt.Errorf("%d: %s", resp.StatusCode, out)
		}
		return nil
	}
	for range *warm {
		if err := do(body(false)); err != nil {
			log.Fatal(err)
		}
	}

	var (
		mu  sync.Mutex
		res []result
		wg  sync.WaitGroup
	)
	send := func(isLong bool, b []byte) {
		defer wg.Done()
		start := time.Now()
		err := do(b)
		mu.Lock()
		res = append(res, result{isLong, time.Since(start), err})
		mu.Unlock()
	}
	start := time.Now()
	if *burst > 0 || *burstLong > 0 {
		for range *burstLong {
			wg.Add(1)
			go send(true, body(true))
		}
		time.Sleep(2 * time.Millisecond)
		for range *burst {
			wg.Add(1)
			go send(false, body(false))
		}
	} else {
		type arrival struct {
			at   time.Duration
			long bool
		}
		var arr []arrival
		rng := rand.New(rand.NewSource(*seed))
		if *rate > 0 {
			for t := time.Duration(0); ; {
				t += time.Duration(rng.ExpFloat64() / *rate * float64(time.Second))
				if t >= *dur {
					break
				}
				arr = append(arr, arrival{t, false})
			}
		}
		if *longEvery > 0 {
			for t := *longEvery / 2; t < *dur; t += *longEvery {
				arr = append(arr, arrival{t, true})
			}
		}
		sort.Slice(arr, func(i, j int) bool { return arr[i].at < arr[j].at })
		for _, a := range arr {
			b := body(a.long)
			time.Sleep(time.Until(start.Add(a.at)))
			wg.Add(1)
			go send(a.long, b)
		}
	}
	wg.Wait()
	wall := time.Since(start)

	for _, isLong := range []bool{false, true} {
		var ds []float64
		errs := 0
		for _, r := range res {
			if r.long != isLong {
				continue
			}
			if r.err != nil {
				errs++
				log.Printf("error: %v", r.err)
				continue
			}
			ds = append(ds, float64(r.d.Microseconds())/1000)
		}
		if len(ds) == 0 {
			continue
		}
		sort.Float64s(ds)
		q := func(p float64) float64 { return ds[min(int(p*float64(len(ds))), len(ds)-1)] }
		mean := 0.0
		for _, d := range ds {
			mean += d
		}
		name := "short"
		if isLong {
			name = "long"
		}
		fmt.Printf("%-5s n=%3d  mean %7.1f  p50 %7.1f  p90 %7.1f  p99 %7.1f  max %7.1f ms  errors %d\n",
			name, len(ds), mean/float64(len(ds)), q(.5), q(.9), q(.99), ds[len(ds)-1], errs)
	}
	fmt.Printf("all   %d requests in %.2f s: %.1f req/s\n", len(res), wall.Seconds(), float64(len(res))/wall.Seconds())
}
