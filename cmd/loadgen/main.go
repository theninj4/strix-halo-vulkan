// Command loadgen is CONCURRENCY.md's C0 harness: it fires voice-shaped and
// agent-shaped chat completions at a running server, concurrently, and
// reports what each stream saw.
//
// A **voice** request is a Home-Assistant-shaped system prompt (a fixed list
// of rooms and devices, ~2k tokens by default) plus a short utterance, with
// thinking off and a short answer. An **agent** request is a long document
// from wikitext plus an instruction that produces a long answer. The agents
// start at t=0 and the voice requests arrive at -voice-at, so a voice command
// lands while the agents are prefilling, decoding, or both.
//
// Each phase is measured against the same streams run **alone**, the same
// hour (-solo, on by default): a concurrent number is only readable next to
// its own control.
//
//	go run ./cmd/serve -llm -llm-ctx 65536 -llm-slots 3 -addr 127.0.0.1:8080 -token= &
//	go run ./cmd/loadgen -url http://127.0.0.1:8080
//
// An **image** request (-image, -image-at) is a user turn holding a picture
// and a short question, the shape of LLM-VISION.md's served vision. Each
// request carries the picture with a counter in a JPEG comment segment, so no
// two share bytes (the pixels are the same) and the server's encoded-image cache never skips the tower: every
// image request pays the tower under the device lock, which is what the
// stall column measures in the other streams.
//
// TTFT is taken at the first delta of any kind (content or reasoning) and the
// decode rate is (completion_tokens - 1) over first-to-last delta, with
// completion_tokens from the usage frame, so a held-back byte or a hidden
// marker moves the edges by a token at most.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"text/tabwriter"
	"time"
)

type stream struct {
	name   string
	class  string // "interactive" or "background": sent as X-Priority
	at     time.Duration
	system string
	user   string
	max    int
	effort string
	image  []byte // a JPEG before the user text; made distinct per request
}

type result struct {
	s             stream
	start         time.Time
	first, last   time.Time
	deltas        int
	times         []time.Time // every delta's arrival
	t0            time.Time   // the phase's start, for stall offsets
	prompt, compl int
	finish        string
	err           error
}

func (r *result) ttft() time.Duration  { return r.first.Sub(r.start) }
func (r *result) total() time.Duration { return r.last.Sub(r.start) }

// gaps are the waits between consecutive deltas longer than min, as
// (offset from the phase start, length).
func (r *result) gaps(min time.Duration) [][2]time.Duration {
	var out [][2]time.Duration
	for i := 1; i < len(r.times); i++ {
		if d := r.times[i].Sub(r.times[i-1]); d > min {
			out = append(out, [2]time.Duration{r.times[i-1].Sub(r.t0), d})
		}
	}
	return out
}

// maxGap is the longest wait between two deltas after the first.
func (r *result) maxGap() time.Duration {
	var m time.Duration
	for i := 1; i < len(r.times); i++ {
		m = max(m, r.times[i].Sub(r.times[i-1]))
	}
	return m
}

// rate is the decode rate the stream saw after its first token.
func (r *result) rate() float64 {
	d := r.last.Sub(r.first).Seconds()
	if r.compl < 2 || d <= 0 {
		return 0
	}
	return float64(r.compl-1) / d
}

func main() {
	url := flag.String("url", "http://127.0.0.1:8080", "server root")
	token := flag.String("token", "", "bearer token, if the server wants one")
	corpus := flag.String("corpus", "models/wikitext-2-raw/wiki.test.raw", "text the agent prompts are cut from")
	agents := flag.Int("agents", 2, "background agent streams, all starting at t=0")
	agentChars := flag.Int("agent-chars", 32000, "characters of document in each agent prompt (~4 a token)")
	agentMax := flag.Int("agent-max", 768, "max_tokens of an agent answer")
	agentClass := flag.String("agent-class", "background", "X-Priority of the agents; empty sends none")
	voiceAt := flag.String("voice-at", "2s,30s", "comma-separated arrival times of the voice requests, from the agents' start")
	voiceDevices := flag.Int("voice-devices", 40, "devices in the voice system prompt (~45 tokens each)")
	voiceMax := flag.Int("voice-max", 32, "max_tokens of a voice answer")
	voiceClass := flag.String("voice-class", "interactive", "X-Priority of the voice requests; empty sends none")
	solo := flag.Bool("solo", true, "run each stream alone first, as the control")
	runs := flag.Int("runs", 1, "times to repeat the concurrent phase")
	imagePath := flag.String("image", "reference/out/llmvision/decode/photo_background.jpg", "picture the image requests carry")
	imageAt := flag.String("image-at", "", "comma-separated arrival times of image requests; empty sends none")
	imageMax := flag.Int("image-max", 64, "max_tokens of an image answer")
	imageClass := flag.String("image-class", "interactive", "X-Priority of the image requests; empty sends none")
	stall := flag.Duration("stall", 300*time.Millisecond, "list every wait between deltas longer than this")
	flag.Parse()

	text, err := os.ReadFile(*corpus)
	if err != nil {
		log.Fatal(err)
	}
	ats := durations("-voice-at", *voiceAt)
	imgAts := durations("-image-at", *imageAt)
	var pic []byte
	if len(imgAts) > 0 {
		if pic, err = os.ReadFile(*imagePath); err != nil {
			log.Fatal(err)
		}
		if !bytes.HasPrefix(pic, []byte{0xff, 0xd8}) {
			log.Fatalf("-image: %s is not a JPEG", *imagePath)
		}
	}

	sys := voiceSystem(*voiceDevices)
	var set []stream
	for i := range *agents {
		off := (i*7919*(*agentChars)/3 + i*104729) % max(1, len(text)-*agentChars)
		doc := string(text[off : off+*agentChars])
		set = append(set, stream{
			name: fmt.Sprintf("agent%d", i), class: *agentClass, max: *agentMax,
			system: "You are a careful research assistant.",
			user: "Here is a document.\n\n" + doc + "\n\nWrite a long, detailed, section-by-section " +
				"summary of everything in the document above, then a list of every person and place it names.",
		})
	}
	for i, at := range ats {
		set = append(set, stream{
			name: fmt.Sprintf("voice%d", i), class: *voiceClass, at: at, max: *voiceMax,
			effort: "none", system: sys, user: utterances[i%len(utterances)],
		})
	}
	for i, at := range imgAts {
		set = append(set, stream{
			name: fmt.Sprintf("image%d", i), class: *imageClass, at: at, max: *imageMax,
			effort: "none", image: pic, user: "Describe this picture in detail.",
		})
	}

	c := &client{url: strings.TrimRight(*url, "/"), token: *token}
	if *solo {
		fmt.Println("== solo: each stream alone ==")
		var rs []*result
		for _, s := range set {
			s.at = 0
			rs = append(rs, c.run(context.Background(), s, time.Now()))
		}
		report(rs, *stall)
	}
	for r := range *runs {
		fmt.Printf("\n== concurrent, run %d: %d agents at t=0, voice at %s ==\n", r+1, *agents, *voiceAt)
		t0 := time.Now()
		rs := make([]*result, len(set))
		var wg sync.WaitGroup
		for i, s := range set {
			wg.Add(1)
			go func() {
				defer wg.Done()
				time.Sleep(time.Until(t0.Add(s.at)))
				rs[i] = c.run(context.Background(), s, t0)
			}()
		}
		wg.Wait()
		report(rs, *stall)
		var tok int
		var end time.Time
		for _, r := range rs {
			tok += r.compl
			if r.last.After(end) {
				end = r.last
			}
		}
		fmt.Printf("aggregate: %d completion tokens in %.1f s = %.2f tok/s wall\n",
			tok, end.Sub(t0).Seconds(), float64(tok)/end.Sub(t0).Seconds())
	}
}

func durations(flagName, list string) []time.Duration {
	var out []time.Duration
	for _, f := range strings.Split(list, ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		d, err := time.ParseDuration(f)
		if err != nil {
			log.Fatalf("%s: %v", flagName, err)
		}
		out = append(out, d)
	}
	return out
}

func report(rs []*result, stall time.Duration) {
	sort.SliceStable(rs, func(i, j int) bool { return rs[i].s.at < rs[j].s.at })
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(w, "stream\tclass\tarrive s\tprompt\tcompl\tTTFT s\tdecode tok/s\tmax gap s\ttotal s\tfinish\t")
	for _, r := range rs {
		if r.err != nil {
			fmt.Fprintf(w, "%s\t%s\t%.1f\terror: %v\t\t\t\t\t\t\t\n", r.s.name, r.s.class, r.s.at.Seconds(), r.err)
			continue
		}
		fmt.Fprintf(w, "%s\t%s\t%.1f\t%d\t%d\t%.3f\t%.2f\t%.3f\t%.2f\t%s\t\n", r.s.name, r.s.class,
			r.s.at.Seconds(), r.prompt, r.compl, r.ttft().Seconds(), r.rate(), r.maxGap().Seconds(),
			r.total().Seconds(), r.finish)
	}
	w.Flush()
	for _, r := range rs {
		for _, g := range r.gaps(stall) {
			fmt.Printf("  stall: %s waited %.3f s from t=%.2f s\n", r.s.name, g[1].Seconds(), g[0].Seconds())
		}
	}
}

type client struct {
	url, token string
}

// seq makes every image request's bytes distinct.
var seq atomic.Uint32

// imagePart is the JPEG s.image as a data URL, with a comment segment holding
// a counter inserted after its SOI marker. Every decoder skips a comment, so
// the pixels are the file's, but the bytes are new and the server's image
// cache (keyed by them) misses. A changed pixel does not do this: JPEG's
// quantization can erase it and give identical bytes.
func imagePart(jpg []byte) map[string]any {
	note := fmt.Sprintf("loadgen %d", seq.Add(1))
	seg := []byte{0xff, 0xfe, byte((len(note) + 2) >> 8), byte(len(note) + 2)}
	b := append(append(append(append([]byte{}, jpg[:2]...), seg...), note...), jpg[2:]...)
	url := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(b)
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
}

func (c *client) run(ctx context.Context, s stream, t0 time.Time) *result {
	r := &result{s: s, t0: t0}
	var messages []map[string]any
	if s.system != "" {
		messages = append(messages, map[string]any{"role": "system", "content": s.system})
	}
	if s.image != nil {
		messages = append(messages, map[string]any{"role": "user", "content": []map[string]any{
			imagePart(s.image), {"type": "text", "text": s.user}}})
	} else {
		messages = append(messages, map[string]any{"role": "user", "content": s.user})
	}
	body := map[string]any{
		"model":          "llm",
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"temperature":    0,
		"max_tokens":     s.max,
		"messages":       messages,
	}
	if s.effort != "" {
		body["reasoning_effort"] = s.effort
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", c.url+"/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		r.err = err
		return r
	}
	req.Header.Set("Content-Type", "application/json")
	if s.class != "" {
		req.Header.Set("X-Priority", s.class)
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	r.start = time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		r.err = err
		return r
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		r.err = fmt.Errorf("%s: %s", resp.Status, bytes.TrimSpace(msg))
		return r
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if line == "[DONE]" {
			break
		}
		var f struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(line), &f); err != nil {
			r.err = err
			return r
		}
		if f.Error != nil {
			r.err = fmt.Errorf("%s", f.Error.Message)
			return r
		}
		if f.Usage != nil {
			r.prompt, r.compl = f.Usage.PromptTokens, f.Usage.CompletionTokens
		}
		for _, ch := range f.Choices {
			if ch.FinishReason != nil {
				r.finish = *ch.FinishReason
			}
			if ch.Delta.Content == "" && ch.Delta.ReasoningContent == "" {
				continue
			}
			now := time.Now()
			r.times = append(r.times, now)
			if r.deltas == 0 {
				r.first = now
			}
			r.last = now
			r.deltas++
		}
	}
	if err := sc.Err(); err != nil {
		r.err = err
	}
	if r.deltas == 0 && r.err == nil {
		r.err = fmt.Errorf("no deltas")
	}
	return r
}

var utterances = []string{
	"Turn on the kitchen lights and set them to fifty percent.",
	"Is the garage door open?",
	"Set the living room thermostat to twenty one degrees.",
	"Turn off every light upstairs.",
}

// voiceSystem is a Home-Assistant-shaped system prompt: instructions, then
// one line per exposed entity. It is deterministic, so every run and every
// request shares it byte for byte.
func voiceSystem(devices int) string {
	rooms := []string{"Kitchen", "Living Room", "Dining Room", "Hallway", "Master Bedroom",
		"Guest Bedroom", "Office", "Bathroom", "Garage", "Garden", "Utility Room", "Nursery",
		"Landing", "Porch", "Loft", "Cellar"}
	kinds := []struct{ domain, name, state string }{
		{"light", "Ceiling Light", "off; brightness: 0; color_temp: 370"},
		{"light", "Lamp", "on; brightness: 128"},
		{"switch", "Socket", "off"},
		{"sensor", "Temperature", "20.5 °C"},
		{"sensor", "Humidity", "48 %"},
		{"binary_sensor", "Motion", "off"},
		{"climate", "Thermostat", "heat; current_temperature: 20.5; temperature: 21"},
		{"cover", "Blind", "open; current_position: 100"},
		{"media_player", "Speaker", "idle; volume_level: 0.3"},
		{"lock", "Door Lock", "locked"},
	}
	var b strings.Builder
	b.WriteString("You are a voice assistant for Home Assistant.\n" +
		"Answer questions about the world truthfully.\n" +
		"Answer in plain text. Keep it simple and to the point.\n" +
		"When controlling Home Assistant always call the intent tools. Use HassTurnOn to lock and " +
		"HassTurnOff to unlock a lock. When controlling a device, prefer passing just name and domain. " +
		"When controlling an area, prefer passing just area name and domain.\n" +
		"When a user asks to turn on all devices of a specific type, ask user to specify an area, " +
		"unless there is only one device of that type.\n" +
		"The current time is 18:42:07. Today's date is 2026-09-23.\n" +
		"An overview of the areas and the devices in this smart home:\n")
	for i := range devices {
		room := rooms[i%len(rooms)]
		k := kinds[(i/len(rooms)+i)%len(kinds)]
		id := strings.ToLower(strings.ReplaceAll(room+"_"+k.name, " ", "_"))
		fmt.Fprintf(&b, "- names: %s %s\n  domain: %s\n  entity_id: %s.%s_%d\n  state: '%s'\n  areas: %s\n",
			room, k.name, k.domain, k.domain, id, i, k.state, room)
	}
	return b.String()
}
