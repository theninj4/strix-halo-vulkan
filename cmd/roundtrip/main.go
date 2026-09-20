// Command roundtrip measures the speech loop through the HTTP API and checks
// that it closes: text -> POST /v1/audio/speech -> resample -> POST
// /v1/audio/transcriptions -> text, and the text that comes back has to be
// the text that went in.
//
// It is a *client*. Nothing here imports kokoro, parakeet or vk, and the two
// legs are measured the way a caller experiences them -- the request's whole
// wall clock, split at the first response byte into what the server spent and
// what the wire did. That split is the point: SPEECH.md's stage tables say an
// utterance is 31 ms and an eleven-second clip transcribes in 43, and this
// says what the endpoints around them cost per second of audio, which is the
// number a user of the API actually waits on.
//
// The server has to be up with both audio models:
//
//	go run ./cmd/serve -tts -stt
//	go run ./cmd/roundtrip
//	go run ./cmd/roundtrip -reps 3 -csv results/roundtrip.csv
//	go run ./cmd/roundtrip -text 'Anything you like.' -v
//
// A resample sits between the legs because the two models do not agree on a
// rate: kokoro emits 24 kHz, parakeet reads 16 kHz, and backend.STT refuses
// the mismatch rather than filtering quietly. So the client filters, and its
// cost is one of the rows in the report -- if it ever stops being a rounding
// error, that is an argument for the server doing it instead.
//
// Exit status is nonzero when a case does not come back, because "the text
// matches" is the correctness bound this benchmark is only meaningful inside:
// a faster loop that mangles the words is not a faster loop.
package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptrace"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"strix-halo-vulkan/audio"
)

// sttRate is what parakeet's front end reads. It is a constant here rather
// than something negotiated because the checkpoint fixes it and the endpoint
// has no field to ask about it -- a client has to know, which is itself worth
// recording in the one program that has to cross the two rates.
const sttRate = 16000

func main() {
	log.SetFlags(0)
	base := flag.String("url", "http://127.0.0.1:8080", "server base URL")
	token := flag.String("token", "womblesofwimbledon", "bearer token; empty sends no Authorization header")
	voice := flag.String("voice", "af_heart", "voice pack, or a comma-joined mixture of them")
	speed := flag.Float64("speed", 1, "speech speed the speech endpoint is asked for")
	reps := flag.Int("reps", 1, "timed repetitions per case; the fastest round trip of each is reported")
	warmup := flag.Bool("warmup", true, "run one untimed round trip first, so a connection setup or a lazy staging is not in the numbers")
	text := flag.String("text", "", "run this one text instead of the built-in corpus")
	cases := flag.String("cases", "", "file of texts, one per line; blank lines and # comments are skipped")
	stress := flag.Bool("stress", false, "add the cases that exercise text normalisation (digits, currency, acronyms); they are reported but never counted as failures")
	csvPath := flag.String("csv", "", "write the per-case rows here, e.g. results/roundtrip.csv")
	verbose := flag.Bool("v", false, "print what came back for every case, not only the ones that differ")
	timeout := flag.Duration("timeout", 2*time.Minute, "per-request timeout")
	flag.Parse()

	cs, err := corpus(*text, *cases, *stress)
	if err != nil {
		log.Fatal(err)
	}

	c := &client{
		base:  strings.TrimRight(*base, "/"),
		token: *token,
		voice: *voice,
		speed: *speed,
		http: &http.Client{
			Timeout: *timeout,
			// One connection, kept: a benchmark that measured a TCP
			// handshake per case would be measuring the loopback.
			Transport: &http.Transport{MaxIdleConnsPerHost: 4, DisableCompression: true},
		},
	}
	if err := c.checkModels(); err != nil {
		log.Fatal(err)
	}

	if *warmup {
		if _, err := c.run(cs[0].text); err != nil {
			log.Fatalf("warmup: %v", err)
		}
	}

	results := make([]result, 0, len(cs))
	for _, cs := range cs {
		var best *result
		for r := 0; r < *reps; r++ {
			got, err := c.run(cs.text)
			if err != nil {
				log.Fatalf("%s: %v", cs.name, err)
			}
			got.name, got.stress, got.sent = cs.name, cs.stress, cs.text
			got.score()
			if best == nil || got.total() < best.total() {
				best = got
			}
		}
		results = append(results, *best)
	}

	report(results, *verbose)
	if *csvPath != "" {
		if err := writeCSV(*csvPath, results); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("\nwrote %s\n", *csvPath)
	}
	for _, r := range results {
		if !r.stress && !r.exact {
			os.Exit(1)
		}
	}
}

// ---------------------------------------------------------------- the corpus

type testCase struct {
	name   string
	text   string
	stress bool
}

// builtin is prose chosen so that a perfect loop returns it exactly. It
// carries no digits, no currency and no acronyms, because those do not
// round-trip through *anything*: kokoro's front end says "twenty twenty four"
// for 2024 and parakeet writes down what it hears, so a mismatch there is two
// correct components disagreeing about spelling rather than a defect. Those
// cases live behind -stress, where they are measured and not judged.
//
// The lengths are the other half of the design. The report fits each leg's
// cost against the audio it handled, and a fit needs a spread: six cases from
// one word to a paragraph separate "this endpoint has fixed overhead" from
// "this endpoint costs what it processes", which are two different things to
// go and fix.
var builtin = []testCase{
	{name: "word", text: "Hello."},
	{name: "sentence", text: "The quick brown fox jumps over the lazy dog."},
	{name: "clause", text: "She sells seashells by the seashore, and the shells she sells are surely seashells."},
	{name: "pair", text: "The engineers agreed that the graphics processor was starved for bandwidth " +
		"long before it ran out of arithmetic, so they rewrote the inner loop."},
	{name: "para", text: "A benchmark that measures the wrong thing is worse than no benchmark at all, " +
		"because it gives the team a number to defend, and a number that must be defended stops " +
		"being a measurement and becomes a promise that nobody wanted to make."},
	{name: "long", text: "Speech synthesis and speech recognition are mirror images that rarely meet " +
		"in the middle. One turns letters into sounds and the other turns sounds back into letters, " +
		"and between them sits a sampling rate that neither model chose, a filter that nobody wanted, " +
		"and a round trip that finally tells the truth about both."},
}

// stressCases are the ones where the two models are each right and still
// disagree. They are here because the disagreement is worth seeing -- it is
// the shape of the work a normaliser would have to do -- and never counted as
// a failure.
var stressCases = []testCase{
	{name: "digits", stress: true, text: "In 2024 the team shipped 1,024 kernels and spent 3.5 million dollars, up 12 percent."},
	{name: "acronym", stress: true, text: "The GPU and the CPU share one 236 gigabyte per second bus on this SoC."},
	{name: "symbols", stress: true, text: "Dr. Patel wrote to the U.S. office at 9 a.m. about the $40 fee."},
}

func corpus(text, path string, stress bool) ([]testCase, error) {
	if text != "" && path != "" {
		return nil, fmt.Errorf("-text and -cases are two ways to say the same thing; pass one")
	}
	if text != "" {
		return []testCase{{name: "arg", text: text}}, nil
	}
	if path != "" {
		buf, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var out []testCase
		for i, line := range strings.Split(string(buf), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			out = append(out, testCase{name: "line" + strconv.Itoa(i+1), text: line})
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("%s holds no cases", path)
		}
		return out, nil
	}
	out := append([]testCase{}, builtin...)
	if stress {
		out = append(out, stressCases...)
	}
	return out, nil
}

// ---------------------------------------------------------------- the client

type client struct {
	base  string
	token string
	voice string
	speed float64
	http  *http.Client
}

// leg is one HTTP request, timed the way a caller sees it.
//
// Server is measured from the moment the request was fully written to the
// first byte of the response, which is the closest a client can get to "what
// the model cost" without the server saying so. Both endpoints buffer their
// whole answer before writing it -- the speech one sets a Content-Length, and
// there is nothing to stream from a vocoder whose durations are decided up
// front (SPEECH.md T2) -- so nothing of the model's time hides in Read.
type leg struct {
	Total  time.Duration
	Write  time.Duration // request accepted by the kernel
	Server time.Duration // written -> first response byte
	Read   time.Duration // first byte -> last byte
	Sent   int           // request body
	Bytes  int           // response body
}

func (c *client) post(path, contentType string, body []byte) (*leg, []byte, http.Header, error) {
	var wrote, first time.Time
	trace := &httptrace.ClientTrace{
		WroteRequest:         func(httptrace.WroteRequestInfo) { wrote = time.Now() },
		GotFirstResponseByte: func() { first = time.Now() },
	}
	req, err := http.NewRequest(http.MethodPost, c.base+path, bytes.NewReader(body))
	if err != nil {
		return nil, nil, nil, err
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	req.Header.Set("Content-Type", contentType)
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	start := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, nil, nil, err
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	done := time.Now()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("reading the response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, nil, nil, fmt.Errorf("%s: %s: %s", path, resp.Status, strings.TrimSpace(string(out)))
	}
	l := &leg{Total: done.Sub(start), Sent: len(body), Bytes: len(out)}
	if !wrote.IsZero() && !first.IsZero() {
		l.Write = wrote.Sub(start)
		l.Server = first.Sub(wrote)
		l.Read = done.Sub(first)
	}
	return l, out, resp.Header, nil
}

// checkModels asks what the process loaded, so a missing flag is one line at
// the start rather than a 501 in the middle of a table.
func (c *client) checkModels() error {
	req, err := http.NewRequest(http.MethodGet, c.base+"/v1/models", nil)
	if err != nil {
		return err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%s is not answering (%v); start it with `go run ./cmd/serve -tts -stt`", c.base, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET /v1/models: %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var models struct {
		Data []struct {
			ID     string   `json:"id"`
			Voices []string `json:"voices"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &models); err != nil {
		return fmt.Errorf("GET /v1/models: %w", err)
	}
	var speech, stt string
	for _, m := range models.Data {
		if len(m.Voices) > 0 {
			speech = m.ID
		}
		if strings.Contains(m.ID, "parakeet") {
			stt = m.ID
		}
	}
	switch {
	case speech == "" && stt == "":
		return fmt.Errorf("%s has neither audio model loaded; restart it with -tts -stt", c.base)
	case speech == "":
		return fmt.Errorf("%s has no speech model; restart it with -tts", c.base)
	case stt == "":
		return fmt.Errorf("%s has no transcription model; restart it with -stt", c.base)
	}
	fmt.Printf("%s: %s -> %s, voice %s\n\n", c.base, speech, stt, c.voice)
	return nil
}

// speak is the first leg. It asks for wav rather than pcm so that the bytes
// the second leg uploads are the bytes the endpoint produced, header and all
// -- a round trip that re-containerised the audio in the middle would be
// measuring this program.
func (c *client) speak(text string) (*leg, *audio.Clip, error) {
	body, err := json.Marshal(map[string]any{
		"input": text, "model": "kokoro", "voice": c.voice,
		"response_format": "wav", "speed": c.speed,
	})
	if err != nil {
		return nil, nil, err
	}
	l, out, _, err := c.post("/v1/audio/speech", "application/json", body)
	if err != nil {
		return nil, nil, err
	}
	clip, err := audio.DecodeWAV(out)
	if err != nil {
		return nil, nil, fmt.Errorf("decoding the speech endpoint's wav: %w", err)
	}
	return l, clip, nil
}

// transcribe is the second leg, as multipart, which is what OpenAI's own
// clients send and therefore the encoding worth timing.
func (c *client) transcribe(clip *audio.Clip) (*leg, string, error) {
	wav, err := clip.EncodeWAV()
	if err != nil {
		return nil, "", err
	}
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	f, err := mw.CreateFormFile("file", "roundtrip.wav")
	if err != nil {
		return nil, "", err
	}
	if _, err := f.Write(wav); err != nil {
		return nil, "", err
	}
	for k, v := range map[string]string{"model": "parakeet", "response_format": "json"} {
		if err := mw.WriteField(k, v); err != nil {
			return nil, "", err
		}
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	l, out, _, err := c.post("/v1/audio/transcriptions", mw.FormDataContentType(), buf.Bytes())
	if err != nil {
		return nil, "", err
	}
	var resp struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		return nil, "", fmt.Errorf("decoding the transcription: %w", err)
	}
	return l, resp.Text, nil
}

// run is one round trip: two requests and the filter between them.
func (c *client) run(text string) (*result, error) {
	r := &result{}
	var err error
	if r.speech, r.clip, err = c.speak(text); err != nil {
		return nil, err
	}
	// The resample is the client's own cost and is timed on its own, because
	// it is the one row in the report that a change to this repository could
	// delete outright by putting it in the server or by giving kokoro's
	// iSTFT head a 16 kHz mode.
	start := time.Now()
	sent := r.clip
	if sent.Rate != sttRate {
		if sent, err = audio.Resample(sent, sttRate); err != nil {
			return nil, err
		}
	}
	r.filter = time.Since(start)
	if r.stt, r.got, err = c.transcribe(sent); err != nil {
		return nil, err
	}
	return r, nil
}

// --------------------------------------------------------------- the results

type result struct {
	name   string
	sent   string
	got    string
	stress bool
	exact  bool
	wer    float64

	clip   *audio.Clip
	speech *leg
	filter time.Duration
	stt    *leg
}

func (r *result) total() time.Duration { return r.speech.Total + r.filter + r.stt.Total }
func (r *result) seconds() float64     { return r.clip.Duration() }

// score compares what came back with what went out, on normalised words.
func (r *result) score() {
	want, got := normalize(r.sent), normalize(r.got)
	r.exact = strings.Join(want, " ") == strings.Join(got, " ")
	r.wer = wordErrorRate(want, got)
}

// normalize is the comparison this benchmark is defined against: case and
// punctuation are the speech endpoint's business rather than the loop's, so a
// transcript that writes "seashore, and" where the input said "seashore and"
// is not a failure of anything measured here. Hyphens separate, so
// "round-trip" and "round trip" are the same two words.
//
// What it does *not* do is normalise numbers or abbreviations. That is a
// deliberate hole: it is exactly the class of disagreement -stress exists to
// show, and a comparison that papered over it would hide the one text
// transformation on this path that nobody has written.
func normalize(s string) []string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
		case r == '\'' || r == '\u2019':
			// Kept, so "don't" stays one word rather than becoming two.
			b.WriteRune('\'')
		default:
			b.WriteRune(' ')
		}
	}
	return strings.Fields(b.String())
}

// wordErrorRate is the usual edit distance over words, over the length of the
// reference. It is here so that a mismatch has a size: one wrong word in a
// paragraph and a transcript that fell apart are both "not exact", and only
// one of them is a bug in the loop.
func wordErrorRate(want, got []string) float64 {
	if len(want) == 0 {
		if len(got) == 0 {
			return 0
		}
		return 1
	}
	prev := make([]int, len(got)+1)
	cur := make([]int, len(got)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(want); i++ {
		cur[0] = i
		for j := 1; j <= len(got); j++ {
			cost := 1
			if want[i-1] == got[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j-1]+cost, min(prev[j]+1, cur[j-1]+1))
		}
		prev, cur = cur, prev
	}
	return float64(prev[len(got)]) / float64(len(want))
}

// ---------------------------------------------------------------- the report

func report(rs []result, verbose bool) {
	fmt.Printf("%-9s %6s %6s %8s  %-17s %7s  %-17s %9s %8s %7s %s\n",
		"case", "chars", "words", "audio", "speech", "->16k", "transcribe", "round trip", "xRT", "WER", "")
	for _, r := range rs {
		mark := "ok"
		switch {
		case r.stress:
			mark = "stress"
		case !r.exact:
			mark = "MISMATCH"
		}
		fmt.Printf("%-9s %6d %6d %7.2fs  %-17s %7s  %-17s %9s %7.1fx %6.1f%% %s\n",
			r.name, len(r.sent), len(strings.Fields(r.sent)), r.seconds(),
			legString(r.speech), ms(r.filter), legString(r.stt),
			ms(r.total()), r.seconds()/r.total().Seconds(), 100*r.wer, mark)
	}

	// Totals, and then the only question the table is asked: which leg is
	// the round trip made of.
	var audioSec float64
	var speech, filter, stt, srvSpeech, srvSTT, wire time.Duration
	var bytesUp, bytesDown int
	for _, r := range rs {
		audioSec += r.seconds()
		speech += r.speech.Total
		filter += r.filter
		stt += r.stt.Total
		srvSpeech += r.speech.Server
		srvSTT += r.stt.Server
		wire += r.speech.Write + r.speech.Read + r.stt.Write + r.stt.Read
		bytesDown += r.speech.Bytes
		bytesUp += r.stt.Sent
	}
	total := speech + filter + stt
	fmt.Printf("\n%.2f s of audio in %v -- the loop runs at %.1fx real time\n",
		audioSec, total.Round(time.Millisecond), audioSec/total.Seconds())

	fmt.Printf("\n%-24s %9s %9s %12s %7s\n", "where the time goes", "total", "ms/s audio", "share", "")
	rows := []struct {
		name string
		d    time.Duration
	}{
		{"speech, server", srvSpeech},
		{"transcribe, server", srvSTT},
		{"resample, client", filter},
		{"http and encoding", wire},
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].d > rows[j].d })
	for _, row := range rows {
		fmt.Printf("%-24s %9s %9.1f %11.1f%%\n", row.name, ms(row.d),
			float64(row.d.Milliseconds())/audioSec, 100*float64(row.d)/float64(total))
	}
	fmt.Printf("\n%-24s %9s %9.1f\n", "speech, whole request", ms(speech), float64(speech.Milliseconds())/audioSec)
	fmt.Printf("%-24s %9s %9.1f\n", "transcribe, whole request", ms(stt), float64(stt.Milliseconds())/audioSec)
	fmt.Printf("%-24s %9s\n", "wav down / up", fmt.Sprintf("%.1f / %.1f MB",
		float64(bytesDown)/1e6, float64(bytesUp)/1e6))

	// The same legs per second of audio, case by case. A total hides the
	// trend, and the trend is the thing worth acting on: a leg whose
	// per-second cost falls as the clip lengthens is amortising a fixed cost
	// and needs nothing, and a leg whose per-second cost *rises* is doing
	// superlinear work and will keep getting worse at the lengths a user
	// actually dictates.
	if plain := nonStress(rs); len(plain) >= 3 {
		legs := []struct {
			name string
			get  func(result) time.Duration
		}{
			{"speech, server", func(r result) time.Duration { return r.speech.Server }},
			{"transcribe, server", func(r result) time.Duration { return r.stt.Server }},
			{"resample, client", func(r result) time.Duration { return r.filter }},
		}
		fmt.Printf("\n%-24s", "ms per second of audio")
		for _, r := range plain {
			fmt.Printf(" %7s", strconv.FormatFloat(r.seconds(), 'f', 2, 64)+"s")
		}
		fmt.Println()
		for _, l := range legs {
			fmt.Printf("%-24s", l.name)
			for _, r := range plain {
				fmt.Printf(" %7.1f", msf(l.get(r))/r.seconds())
			}
			fmt.Println()
		}
	}

	// Fixed cost against marginal cost, which is the thing the table cannot
	// be read for: a leg that is all intercept wants its protocol looked at,
	// and a leg that is all slope wants its arithmetic looked at.
	if len(rs) >= 3 {
		fmt.Printf("\n%-24s %12s %14s %8s\n", "cost against audio", "fixed", "per second", "r2")
		for _, l := range []struct {
			name string
			get  func(result) time.Duration
		}{
			{"speech, server", func(r result) time.Duration { return r.speech.Server }},
			{"transcribe, server", func(r result) time.Duration { return r.stt.Server }},
			{"resample, client", func(r result) time.Duration { return r.filter }},
			{"round trip", func(r result) time.Duration { return r.total() }},
		} {
			plain := nonStress(rs)
			xs := make([]float64, 0, len(plain))
			ys := make([]float64, 0, len(plain))
			for _, r := range plain {
				xs = append(xs, r.seconds())
				ys = append(ys, msf(l.get(r)))
			}
			a, b, r2 := fit(xs, ys)
			fmt.Printf("%-24s %9.1f ms %11.1f ms %8.3f\n", l.name, a, b, r2)
		}
	}

	for _, r := range rs {
		if verbose || (!r.exact && !r.stress) || (r.stress && !r.exact) {
			fmt.Printf("\n%s:\n  sent %q\n  got  %q\n", r.name, r.sent, r.got)
			if !r.exact {
				if d := diff(normalize(r.sent), normalize(r.got)); d != "" {
					fmt.Printf("  diff %s\n", d)
				}
			}
		}
	}
}

// nonStress is the cases the fits and the per-second table are computed over.
// A normalisation case is a legitimate measurement of the endpoints and a
// misleading point on a line -- its text is short and its audio is long,
// because a front end that says "one thousand and twenty four" for 1,024
// speaks for longer than the characters suggest.
func nonStress(rs []result) []result {
	out := make([]result, 0, len(rs))
	for _, r := range rs {
		if !r.stress {
			out = append(out, r)
		}
	}
	return out
}

// fit is an ordinary least squares y = a + b*x, with the coefficient of
// determination so a straight line that is not one is visible as such.
func fit(xs, ys []float64) (a, b, r2 float64) {
	n := float64(len(xs))
	if n < 2 {
		return 0, 0, 0
	}
	var sx, sy, sxx, sxy float64
	for i := range xs {
		sx, sy = sx+xs[i], sy+ys[i]
		sxx, sxy = sxx+xs[i]*xs[i], sxy+xs[i]*ys[i]
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return sy / n, 0, 0
	}
	b = (n*sxy - sx*sy) / den
	a = (sy - b*sx) / n
	mean := sy / n
	var ssTot, ssRes float64
	for i := range xs {
		ssTot += (ys[i] - mean) * (ys[i] - mean)
		ssRes += (ys[i] - (a + b*xs[i])) * (ys[i] - (a + b*xs[i]))
	}
	if ssTot == 0 {
		return a, b, 1
	}
	return a, b, 1 - ssRes/ssTot
}

// diff names the first few words that differ, which is what tells a
// normalisation disagreement ("2024" against "twenty twenty four") apart from
// a loop that broke.
func diff(want, got []string) string {
	var out []string
	for i := 0; i < len(want) || i < len(got); i++ {
		var w, g string
		if i < len(want) {
			w = want[i]
		}
		if i < len(got) {
			g = got[i]
		}
		if w != g {
			out = append(out, fmt.Sprintf("[%d] %q != %q", i, w, g))
			if len(out) == 4 {
				return strings.Join(out, ", ") + ", ..."
			}
		}
	}
	return strings.Join(out, ", ")
}

func legString(l *leg) string {
	return fmt.Sprintf("%7s (%s srv)", ms(l.Total), strings.TrimSpace(ms(l.Server)))
}

func ms(d time.Duration) string {
	if d >= time.Second {
		return strconv.FormatFloat(d.Seconds(), 'f', 2, 64) + "s"
	}
	return strconv.FormatInt(int64(math.Round(float64(d)/float64(time.Millisecond))), 10) + "ms"
}

func writeCSV(path string, rs []result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	head := []string{
		"case", "stress", "chars", "words", "audio_s", "audio_rate",
		"speech_ms", "speech_server_ms", "speech_read_ms", "speech_wav_bytes",
		"filter_ms", "stt_ms", "stt_server_ms", "stt_write_ms", "stt_upload_bytes",
		"total_ms", "xrt", "wer", "exact",
	}
	if err := w.Write(head); err != nil {
		return err
	}
	for _, r := range rs {
		row := []string{
			r.name, strconv.FormatBool(r.stress),
			strconv.Itoa(len(r.sent)), strconv.Itoa(len(strings.Fields(r.sent))),
			f3(r.seconds()), strconv.Itoa(r.clip.Rate),
			f3(msf(r.speech.Total)), f3(msf(r.speech.Server)), f3(msf(r.speech.Read)),
			strconv.Itoa(r.speech.Bytes),
			f3(msf(r.filter)),
			f3(msf(r.stt.Total)), f3(msf(r.stt.Server)), f3(msf(r.stt.Write)),
			strconv.Itoa(r.stt.Sent),
			f3(msf(r.total())), f3(r.seconds() / r.total().Seconds()), f3(r.wer),
			strconv.FormatBool(r.exact),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}

func msf(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func f3(v float64) string         { return strconv.FormatFloat(v, 'f', 3, 64) }
