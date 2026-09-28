package backend

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"strix-halo-vulkan/api"
)

// TestPickEmbed pins the pass rules: round-robin across requests in arrival
// order, the row budget, a request passed over (not the end of the pass)
// when its next input does not fit, and an over-budget input run alone.
func TestPickEmbed(t *testing.T) {
	always := func([]int) bool { return true }
	cases := []struct {
		name   string
		lens   [][]int
		next   []int
		budget int
		fits   func([]int) bool
		want   []embedTake
	}{
		{"a lone query rides with a big job", [][]int{{10, 10, 10, 10}, {5}}, []int{0, 0}, 25, always,
			[]embedTake{{0, 0}, {1, 0}, {0, 1}}},
		{"resumes where the last pass stopped", [][]int{{10, 10, 10, 10}}, []int{3}, 100, always,
			[]embedTake{{0, 3}}},
		{"a long input does not block the short ones", [][]int{{90}, {5, 5}}, []int{0, 0}, 50, always,
			[]embedTake{{1, 0}, {1, 1}}},
		{"over the budget runs alone", [][]int{{900}, {5}}, []int{0, 1}, 50, always,
			[]embedTake{{0, 0}}},
		{"the encoder's fit is respected", [][]int{{4, 4, 4, 4}}, []int{0}, 100,
			func(l []int) bool { return len(l) <= 2 }, []embedTake{{0, 0}, {0, 1}}},
	}
	for _, c := range cases {
		got := pickEmbed(c.lens, c.next, c.budget, c.fits)
		if !slices.Equal(got, c.want) {
			t.Errorf("%s: picked %v, want %v", c.name, got, c.want)
		}
	}
}

// TestEmbedConcurrent is the served shape of E7: requests arriving together
// share passes, and every vector is the one its text gets on its own.
func TestEmbedConcurrent(t *testing.T) {
	const model = "../models/Qwen3-Embedding-0.6B"
	if _, err := os.Stat(model + "/model.safetensors"); err != nil {
		t.Skipf("no checkpoint at %s", model)
	}
	dev, err := OpenDevice("embed-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer dev.Close()
	b, err := NewEmbed(EmbedOptions{Model: model, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	words := strings.Fields("memory bandwidth caches kernels a quick brown fox jumps over lazy dogs in vulkan compute")
	text := func(r, i int) string {
		var s []string
		for w := 0; w < 3+(r*5+i*3)%40; w++ {
			s = append(s, words[(w+r+i)%len(words)])
		}
		return strings.Join(s, " ")
	}
	sizes := []int{1, 32, 1, 128, 5, 1, 1, 64}
	want := make([][][]float32, len(sizes))
	for r, n := range sizes {
		want[r] = make([][]float32, n)
		for i := 0; i < n; i++ {
			res, err := b.Embed(context.Background(), &api.EmbeddingRequest{Input: []string{text(r, i)}})
			if err != nil {
				t.Fatal(err)
			}
			want[r][i] = res.Vectors[0]
		}
	}
	before := b.passes

	var wg sync.WaitGroup
	errs := make([]error, len(sizes))
	took := make([]time.Duration, len(sizes))
	start := time.Now()
	for r, n := range sizes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			in := make([]string, n)
			for i := range in {
				in[i] = text(r, i)
			}
			res, err := b.Embed(context.Background(), &api.EmbeddingRequest{Input: in})
			took[r] = time.Since(start)
			if err != nil {
				errs[r] = err
				return
			}
			// Batched against lone: the plans differ with the pass width, so
			// this is a cosine, not bits (TestGPUBatchMatchesSingle has the
			// bits under one plan).
			for i, v := range res.Vectors {
				var dot float64
				for e := range v {
					dot += float64(v[e]) * float64(want[r][i][e])
				}
				if 1-dot > 1e-5 {
					errs[r] = fmt.Errorf("request %d input %d: cosine %.7f to its lone vector", r, i, dot)
					return
				}
			}
		}()
	}
	wg.Wait()
	all := time.Since(start)
	for _, err := range errs {
		if err != nil {
			t.Error(err)
		}
	}
	total := 0
	for _, n := range sizes {
		total += n
	}
	t.Logf("%d requests, %d inputs, in %d passes and %v; each request done at %v",
		len(sizes), total, b.passes-before, all.Round(time.Millisecond), took)
	if b.passes-before >= total/4 {
		t.Errorf("%d inputs took %d passes; they are not being batched", total, b.passes-before)
	}
}

// TestEmbedCancel checks a client that hangs up gets its error and leaves
// nothing behind that stops the next request.
func TestEmbedCancel(t *testing.T) {
	const model = "../models/Qwen3-Embedding-0.6B"
	if _, err := os.Stat(model + "/model.safetensors"); err != nil {
		t.Skipf("no checkpoint at %s", model)
	}
	dev, err := OpenDevice("embed-cancel-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer dev.Close()
	b, err := NewEmbed(EmbedOptions{Model: model, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	in := make([]string, 400)
	for i := range in {
		in[i] = strings.Repeat("a long document about nothing in particular ", 8) + fmt.Sprint(i)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := b.Embed(ctx, &api.EmbeddingRequest{Input: in}); err == nil {
		t.Fatal("a 400-input request finished inside 30 ms; the cancellation was not exercised")
	}
	res, err := b.Embed(context.Background(), &api.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil || len(res.Vectors) != 1 {
		t.Fatalf("the request after a cancelled one: %v", err)
	}
	b.qmu.Lock()
	defer b.qmu.Unlock()
	if len(b.queue) != 0 {
		t.Errorf("%d jobs left in the queue", len(b.queue))
	}
}

// TestEmbedQueryOvertakesJob is the head-of-line check: a one-text query
// that arrives while a large request is running joins the next pass and
// comes back long before the large request does. It once came back *with*
// it: the request path took the model lock a pass holds, so the query could
// not queue until the passes ahead of it had drained.
func TestEmbedQueryOvertakesJob(t *testing.T) {
	const model = "../models/Qwen3-Embedding-0.6B"
	if _, err := os.Stat(model + "/model.safetensors"); err != nil {
		t.Skipf("no checkpoint at %s", model)
	}
	dev, err := OpenDevice("embed-hol-test")
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	defer dev.Close()
	b, err := NewEmbed(EmbedOptions{Model: model, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	in := make([]string, 256)
	for i := range in {
		in[i] = strings.Repeat("a paragraph about bandwidth and caches ", 5) + fmt.Sprint(i)
	}
	jobDone := make(chan time.Time, 1)
	start := time.Now()
	go func() {
		if _, err := b.Embed(context.Background(), &api.EmbeddingRequest{Input: in}); err != nil {
			t.Error(err)
		}
		jobDone <- time.Now()
	}()
	time.Sleep(40 * time.Millisecond)
	q := time.Now()
	if _, err := b.Embed(context.Background(), &api.EmbeddingRequest{Input: []string{"hello"}}); err != nil {
		t.Fatal(err)
	}
	query := time.Since(q)
	job := (<-jobDone).Sub(start)
	t.Logf("query %v behind a %d-text job of %v", query.Round(time.Millisecond), len(in), job.Round(time.Millisecond))
	if query > job/3 {
		t.Errorf("the query took %v of the job's %v; it is waiting behind the job", query, job)
	}
}
