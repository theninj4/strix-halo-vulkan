// Command acecal builds and scores MUSIC.md A14e's calibration set: the
// normal songs of set.json (each at every seed), then the known-bad
// controls made from them, every one through the auto score. Each item
// leaves <id>.mp3 (to listen to), <id>.flac and <id>.lat (for the controls
// built on it) and <id>.json, its record; an item whose record exists is
// read back, not redone, so a stopped run resumes.
//
//	go run ./cmd/acecal -out out/a14e
//	go run ./cmd/acecal -out out/a14e -loops   # add ace/loops to the records, no device
//
// A record holds the score as generated ("quality"), the score again
// through pipeline.Rescore ("rescore", which for a song must reproduce
// "quality", and for a swap is the score against the other song's caption
// or lyrics), and the LM's half over codes re-tokenized from the render
// ("render", A-o16) beside the plan's.
//
// The controls:
//   - steps: the song's request at 1 or 2 DiT steps;
//   - swap-caption, swap-lyrics: a song scored against another's caption
//     or lyrics;
//   - lyric-loop: the song's request with one line sung over and over;
//   - audio-loop: a few seconds from a quarter of the way into a song,
//     tiled to its length, scored against its request and plan;
//   - cover0: a cover of a song at audio_cover_strength 0 (the source
//     ignored), scored on its render's codes;
//   - noise, silence: white noise or silence at a song's length, scored
//     against its request and plan.
package main

import (
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/loops"
	"strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/backend"
	"strix-halo-vulkan/vk"
)

//go:embed set.json
var setJSON []byte

type song struct {
	ID, Caption, Lyrics, Lang string
	Duration                  float64
}

type control struct {
	ID, Kind, Of, From, Line string
	Steps, Times             int
	Seconds                  float64
}

type set struct {
	Seeds    []uint64
	Songs    []song
	Controls []control
}

// record is one item's result, as <id>.json.
type record struct {
	ID      string `json:"id"`
	Kind    string `json:"kind"` // "song" or a control's kind
	Of      string `json:"of,omitempty"`
	From    string `json:"from,omitempty"`
	Control bool   `json:"control"`
	Seed    uint64 `json:"seed"`
	Steps   int    `json:"steps,omitempty"`

	Request *plan.Request `json:"request"`
	DiT     *plan.Request `json:"dit"`
	Meta    lm.Meta       `json:"meta,omitempty"`
	CoT     string        `json:"cot,omitempty"`
	Codes   []int         `json:"codes,omitempty"` // the plan's
	Seconds float64       `json:"seconds"`

	Quality *pipeline.Quality `json:"quality,omitempty"`
	Rescore *pipeline.Quality `json:"rescore,omitempty"`
	Render  *lm.Score         `json:"render,omitempty"`
	// NonFinite lists the scores that came out NaN or ±Inf, stored as null.
	NonFinite []string `json:"non_finite,omitempty"`
	Took      float64  `json:"took_s"`
}

// item is a record with the audio the controls are built from.
type item struct {
	rec *record
	wav []float32
	lat []float32
}

func main() {
	models := flag.String("models", "models", "checkpoint root")
	out := flag.String("out", "out/a14e", "output directory")
	only := flag.String("only", "", "comma-separated ids to run (and what they need); empty runs the set")
	loopsOnly := flag.Bool("loops", false, "add the loop measure (ace/loops) to every record already in -out, from its flac, and stop: no device")
	flag.Parse()
	if *loopsOnly {
		if err := addLoops(*out); err != nil {
			log.Fatal(err)
		}
		return
	}

	var s set
	if err := json.Unmarshal(setJSON, &s); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	want := map[string]bool{}
	for _, id := range strings.Split(*only, ",") {
		if id != "" {
			want[id] = true
		}
	}
	run := func(id string) bool { return len(want) == 0 || want[id] }

	dev, err := backend.OpenDevice("ace")
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	err = dev.Do(func(d *vk.Device) error {
		ceiling := 0.0
		for _, sg := range s.Songs {
			ceiling = max(ceiling, sg.Duration)
		}
		p, err := pipeline.New(d, pipeline.DefaultDirs(*models), ceiling)
		if err != nil {
			return err
		}
		defer p.Destroy()
		if err := p.LoadLM(d, pipeline.LMDir(*models), lm.DefaultOptions()); err != nil {
			return err
		}
		if err := p.LoadAesthetics(d, pipeline.AesDir(*models)); err != nil {
			return err
		}
		if err := p.LoadSongEval(d, *models, ceiling); err != nil {
			return err
		}
		c := &cal{p: p, dir: *out, items: map[string]*item{}, songs: map[string]song{}}
		for _, sg := range s.Songs {
			c.songs[sg.ID] = sg
		}
		need := map[string]bool{}
		for _, ct := range s.Controls {
			if run(ct.ID) {
				need[ct.Of] = true
			}
		}
		for _, sg := range s.Songs {
			for _, seed := range s.Seeds {
				id := fmt.Sprintf("%s-%d", sg.ID, seed)
				if run(id) || run(sg.ID) || need[id] {
					if err := c.do(id, func() (*item, error) { return c.song(id, sg, seed) }); err != nil {
						return err
					}
				}
			}
		}
		for _, ct := range s.Controls {
			if run(ct.ID) {
				if err := c.do(ct.ID, func() (*item, error) { return c.control(ct, s.Seeds[0]) }); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}

type cal struct {
	p     *pipeline.Pipeline
	dir   string
	items map[string]*item
	songs map[string]song
}

func (c *cal) path(id, ext string) string { return filepath.Join(c.dir, id+ext) }

// do runs one item, or reads it back when its record is already there.
func (c *cal) do(id string, f func() (*item, error)) error {
	if it, err := c.load(id); err == nil {
		c.items[id] = it
		log.Printf("%s: done before", id)
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	t0 := time.Now()
	it, err := f()
	if err != nil {
		return fmt.Errorf("%s: %w", id, err)
	}
	it.rec.Took = time.Since(t0).Seconds()
	if err := c.save(it); err != nil {
		return err
	}
	c.items[id] = it
	log.Printf("%s: %.0f s of audio in %.1f s: %s", id, it.rec.Seconds, it.rec.Took, summary(it.rec))
	return nil
}

func (c *cal) save(it *item) error {
	r := it.rec
	if err := pipeline.Write(c.path(r.ID, ".mp3"), it.wav); err != nil {
		return err
	}
	if err := pipeline.Write(c.path(r.ID, ".flac"), it.wav); err != nil {
		return err
	}
	if err := writeF32(c.path(r.ID, ".lat"), it.lat); err != nil {
		return err
	}
	r.NonFinite = nil
	nullNonFinite(reflect.ValueOf(r), "", &r.NonFinite)
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.path(r.ID, ".json"), b, 0o644)
}

func (c *cal) load(id string) (*item, error) {
	b, err := os.ReadFile(c.path(id, ".json"))
	if err != nil {
		return nil, err
	}
	it := &item{rec: &record{}}
	if err := json.Unmarshal(b, it.rec); err != nil {
		return nil, err
	}
	if it.wav, err = pipeline.ReadAudio(c.path(id, ".flac")); err != nil {
		return nil, err
	}
	if it.lat, err = readF32(c.path(id, ".lat")); err != nil {
		return nil, err
	}
	return it, nil
}

func request(sg song) *plan.Request {
	return &plan.Request{Caption: sg.Caption, Lyrics: sg.Lyrics, Duration: sg.Duration, Language: sg.Lang}
}

// generate is one song on the thinking path (cmd/ace's defaults), rescored.
func (c *cal) generate(rec *record, r *plan.Request, o pipeline.Options) (*item, error) {
	o.Think, o.Score = true, true
	o.LMSeed = o.Seed
	res, err := c.p.Generate(r, o)
	if err != nil {
		return nil, err
	}
	rec.Request, rec.DiT, rec.Meta, rec.CoT, rec.Codes = r, res.DiT, res.Meta, res.CoT, res.Codes
	rec.Seconds, rec.Quality, rec.Seed = res.Seconds, res.Quality, o.Seed
	it := &item{rec: rec, wav: res.Audio, lat: res.Latents}
	rec.Rescore, rec.Render, err = c.p.Rescore(r, res.DiT, res.Meta, res.Audio, res.Latents, res.Codes, o.Seed)
	return it, err
}

func (c *cal) song(id string, sg song, seed uint64) (*item, error) {
	return c.generate(&record{ID: id, Kind: "song"}, request(sg), pipeline.Options{Seed: seed})
}

// source is the item a control is built on.
func (c *cal) source(id string) (*item, error) {
	it := c.items[id]
	if it == nil {
		return nil, fmt.Errorf("no item %q", id)
	}
	return it, nil
}

func (c *cal) control(ct control, seed uint64) (*item, error) {
	rec := &record{ID: ct.ID, Kind: ct.Kind, Of: ct.Of, From: ct.From, Control: true, Seed: seed}
	switch ct.Kind {
	case "steps":
		sg, ok := c.songs[ct.Of]
		if !ok {
			return nil, fmt.Errorf("no song %q", ct.Of)
		}
		rec.Steps = ct.Steps
		ts := plan.ValidTimesteps[:1] // 1.0
		if ct.Steps == 2 {
			ts = []float64{1, 0.5}
		} else if ct.Steps != 1 {
			return nil, fmt.Errorf("steps %d: only 1 or 2", ct.Steps)
		}
		return c.generate(rec, request(sg), pipeline.Options{Seed: seed, Timesteps: ts})
	case "lyric-loop":
		sg, ok := c.songs[ct.Of]
		if !ok {
			return nil, fmt.Errorf("no song %q", ct.Of)
		}
		lines := []string{"[Verse]"}
		for range ct.Times {
			lines = append(lines, ct.Line)
		}
		r := request(sg)
		r.Lyrics = strings.Join(lines, "\n")
		return c.generate(rec, r, pipeline.Options{Seed: seed})
	case "cover0":
		src, err := c.source(ct.Of)
		if err != nil {
			return nil, err
		}
		a := pipeline.NewAudio("cover")
		a.Source, a.CoverStrength = src.wav, 0
		if err := a.Check(); err != nil {
			return nil, err
		}
		r := *src.rec.Request
		res, err := c.p.Generate(&r, pipeline.Options{Seed: seed, Audio: a, Score: true})
		if err != nil {
			return nil, err
		}
		rec.Request, rec.DiT, rec.Seconds, rec.Quality = &r, res.DiT, res.Seconds, res.Quality
		it := &item{rec: rec, wav: res.Audio, lat: res.Latents}
		rec.Rescore, rec.Render, err = c.p.Rescore(&r, res.DiT, nil, res.Audio, res.Latents, nil, seed)
		return it, err
	}

	// The rest score a song's audio, or audio made in its place, against a
	// request: the song's own or a swap's.
	src, err := c.source(ct.Of)
	if err != nil {
		return nil, err
	}
	s := src.rec
	r, d := *s.Request, *s.DiT
	wav, lat := src.wav, src.lat
	switch ct.Kind {
	case "swap-caption", "swap-lyrics":
		from, err := c.source(ct.From + "-" + fmt.Sprint(seed))
		if err != nil {
			return nil, err
		}
		if ct.Kind == "swap-caption" {
			// The caption the other song's DiT got: its CoT's rewrite
			// when phase 1 wrote one.
			r.Caption, d.Caption = from.rec.Request.Caption, from.rec.DiT.Caption
		} else {
			r.Lyrics, d.Lyrics = from.rec.Request.Lyrics, from.rec.DiT.Lyrics
		}
	case "audio-loop":
		n := len(wav)
		at := n / 4 / 2 * 2
		clip := wav[at : at+int(ct.Seconds*plan.SampleRate)*2]
		wav = make([]float32, n)
		for i := 0; i < n; i += len(clip) {
			copy(wav[i:], clip)
		}
		lat = nil
	case "noise":
		rng := rand.New(rand.NewPCG(seed, 0x0153))
		wav = make([]float32, len(wav))
		for i := range wav {
			wav[i] = float32(rng.NormFloat64() * 0.25)
		}
		lat = nil
	case "silence":
		wav, lat = make([]float32, len(wav)), nil
	default:
		return nil, fmt.Errorf("unknown control kind %q", ct.Kind)
	}
	rec.Request, rec.DiT, rec.Meta, rec.CoT, rec.Codes, rec.Seconds = &r, &d, s.Meta, s.CoT, s.Codes, s.Seconds
	rec.Rescore, rec.Render, err = c.p.Rescore(&r, &d, s.Meta, wav, lat, s.Codes, seed)
	if err != nil {
		return nil, err
	}
	if lat == nil {
		lat = []float32{} // the encode's latents are not kept
	}
	return &item{rec: rec, wav: wav, lat: lat}, nil
}

// addLoops measures every record's audio with ace/loops and writes the
// result into its scores.
func addLoops(dir string) error {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return err
	}
	c := &cal{dir: dir}
	for _, p := range paths {
		id := strings.TrimSuffix(filepath.Base(p), ".json")
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		var rec record
		if json.Unmarshal(b, &rec) != nil || rec.ID != id {
			continue // not a record (panel.json, ratings.json)
		}
		wav, err := pipeline.ReadAudio(c.path(id, ".flac"))
		if err != nil {
			return err
		}
		l := loops.Measure(wav)
		for _, q := range []*pipeline.Quality{rec.Quality, rec.Rescore} {
			if q != nil {
				q.Loops = &l
			}
		}
		if b, err = json.MarshalIndent(&rec, "", "  "); err != nil {
			return err
		}
		if err := os.WriteFile(p, b, 0o644); err != nil {
			return err
		}
		log.Printf("%s: repetition %.3f at %.1f s, loop %.1f s (%.2f) at %.1f s, %v", id, l.Repetition, l.RepetitionLag,
			l.LoopSeconds, l.LoopShare, l.LoopLag, l.Took.Round(time.Millisecond))
	}
	return nil
}

// summary is a record's headline numbers for the log.
func summary(r *record) string {
	q := r.Rescore
	if q == nil {
		return "no score"
	}
	var b strings.Builder
	if q.LM != nil {
		fmt.Fprintf(&b, "reward %.3f", q.LM.Reward)
		if r.Render != nil {
			fmt.Fprintf(&b, " (render %.3f)", r.Render.Reward)
		}
	}
	if q.Lyrics != nil {
		fmt.Fprintf(&b, ", align %.3f/%.3f", q.Lyrics.LM.Score, q.Lyrics.DiT.Score)
	}
	if a := q.Aesthetics; a != nil {
		fmt.Fprintf(&b, ", PQ %.2f CE %.2f", a.PQ, a.CE)
	}
	if s := q.Song; s != nil {
		fmt.Fprintf(&b, ", musicality %.2f", s.Musicality)
	}
	if g := r.Quality; g != nil && g.LM != nil && q.LM != nil && r.Kind == "song" {
		fmt.Fprintf(&b, "; rescore reward moved %.2g", q.LM.Reward-g.LM.Reward)
	}
	return b.String()
}

// nullNonFinite sets every NaN or infinite float64 under v to zero and
// lists its path, since JSON has no NaN; a reader treats a listed path as
// missing.
func nullNonFinite(v reflect.Value, at string, bad *[]string) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			nullNonFinite(v.Elem(), at, bad)
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if v.Type().Field(i).IsExported() {
				nullNonFinite(v.Field(i), at+"."+v.Type().Field(i).Name, bad)
			}
		}
	case reflect.Map:
		if v.Type().Elem().Kind() == reflect.Float64 {
			for _, k := range v.MapKeys() {
				if f := v.MapIndex(k).Float(); math.IsNaN(f) || math.IsInf(f, 0) {
					v.SetMapIndex(k, reflect.ValueOf(0.0))
					*bad = append(*bad, fmt.Sprintf("%s[%v]", at, k))
				}
			}
		}
	case reflect.Float64:
		if f := v.Float(); (math.IsNaN(f) || math.IsInf(f, 0)) && v.CanSet() {
			v.SetFloat(0)
			*bad = append(*bad, at)
		}
	}
}

func writeF32(path string, x []float32) error {
	b := make([]byte, 4*len(x))
	for i, f := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return os.WriteFile(path, b, 0o644)
}

func readF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	x := make([]float32, len(b)/4)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return x, nil
}
