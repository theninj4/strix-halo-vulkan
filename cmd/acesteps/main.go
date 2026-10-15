// Command acesteps is research/music-vertical.md A15 S3's sweep: the base
// model's step budget priced by the auto score. Every song of A14e's set
// (cmd/acecal/set.json, one seed) is planned once by the LM on the base
// model and rendered by each arm from the same codes and the same noise,
// and every render goes through the whole auto score (A14): Audiobox
// Aesthetics, SongEval, the LM's verdict over the render's codes, the DiT's
// lyric alignment and the loop measure. The turbo arm renders the same
// plans on the turbo checkpoint, for the pivot's own comparison.
//
// Each render leaves <song>-<arm>.mp3 (to listen to) and .json (its record);
// a record that exists is read back, not redone, so a stopped run resumes.
// The plans are kept as <song>.plan.json. The summary is per arm: the mean
// of every component, the mean paired difference from b32, and the DiT's
// time.
//
//	go run ./cmd/acesteps -out out/a15/steps
//	go run ./cmd/acesteps -out out/a15/steps -arms b32,b24 -only n01,n02
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"strix-halo-vulkan/ace/lm"
	"strix-halo-vulkan/ace/pipeline"
	"strix-halo-vulkan/ace/plan"
	"strix-halo-vulkan/backend"
	"strix-halo-vulkan/vk"
)

type song struct {
	ID       string  `json:"id"`
	Caption  string  `json:"caption"`
	Lyrics   string  `json:"lyrics"`
	Lang     string  `json:"lang"`
	Duration float64 `json:"duration"`
}

// arm is one way of rendering a plan.
type arm struct {
	name     string
	turbo    bool
	steps    int
	cfgStart float64
}

var arms = map[string]arm{
	"b32":    {name: "b32", steps: 32},
	"b24":    {name: "b24", steps: 24},
	"b16":    {name: "b16", steps: 16},
	"b12":    {name: "b12", steps: 12},
	"b32c50": {name: "b32c50", steps: 32, cfgStart: 0.5}, // CFG only while t ≥ 0.5
	"b24c50": {name: "b24c50", steps: 24, cfgStart: 0.5},
	"turbo":  {name: "turbo", turbo: true},
}

// savedPlan is a song's LM plan, as Render takes it.
type savedPlan struct {
	Meta  lm.Meta       `json:"meta"`
	CoT   string        `json:"cot"`
	Codes []int         `json:"codes"`
	DiT   *plan.Request `json:"dit"`
}

// record is one render's result.
type record struct {
	Song     string             `json:"song"`
	Arm      string             `json:"arm"`
	Seconds  float64            `json:"seconds"`
	DiT      float64            `json:"dit_seconds"`
	Forwards int                `json:"forwards"`
	Metrics  map[string]float64 `json:"metrics"`
}

func main() {
	models := flag.String("models", "models", "checkpoint root")
	setPath := flag.String("set", "cmd/acecal/set.json", "the song set")
	out := flag.String("out", "out/a15/steps", "output directory")
	armList := flag.String("arms", "b32,b24,b16,b32c50,turbo", "arms to render, comma-separated (b32 always plans)")
	only := flag.String("only", "", "comma-separated song ids (empty: the whole set)")
	seed := flag.Uint64("seed", 11, "the DiT's noise seed and the LM's sampling seed")
	flag.Parse()

	raw, err := os.ReadFile(*setPath)
	if err != nil {
		log.Fatal(err)
	}
	var set struct {
		Songs []song `json:"songs"`
	}
	if err := json.Unmarshal(raw, &set); err != nil {
		log.Fatal(err)
	}
	songs := set.Songs
	if *only != "" {
		keep := map[string]bool{}
		for _, id := range strings.Split(*only, ",") {
			keep[id] = true
		}
		songs = nil
		for _, s := range set.Songs {
			if keep[s.ID] {
				songs = append(songs, s)
			}
		}
	}
	var base, turbo []arm
	for _, n := range strings.Split(*armList, ",") {
		a, ok := arms[n]
		if !ok {
			log.Fatalf("no arm %q", n)
		}
		if a.turbo {
			turbo = append(turbo, a)
		} else if n != "b32" {
			base = append(base, a)
		}
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	dev, err := backend.OpenDevice("acesteps")
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	for _, pass := range []struct {
		dit  string
		arms []arm
	}{{"base", base}, {"turbo", turbo}} {
		if pass.dit == "turbo" && len(turbo) == 0 {
			continue
		}
		err := dev.Do(func(d *vk.Device) error {
			p, err := load(d, *models, pass.dit)
			if err != nil {
				return err
			}
			defer p.Destroy()
			for _, s := range songs {
				if err := run(p, *out, s, pass.dit, pass.arms, *seed); err != nil {
					return fmt.Errorf("%s: %w", s.ID, err)
				}
			}
			return nil
		})
		if err != nil {
			log.Fatal(err)
		}
	}
	summarise(*out, songs, *armList)
}

// load stages a pipeline on the named DiT with the LM and every scorer.
func load(d *vk.Device, models, dit string) (*pipeline.Pipeline, error) {
	dirs, err := pipeline.DirsFor(models, dit)
	if err != nil {
		return nil, err
	}
	p, err := pipeline.New(d, dirs, plan.MaxSeconds)
	if err != nil {
		return nil, err
	}
	if err := p.LoadLM(d, pipeline.LMDir(models), lm.DefaultOptions()); err != nil {
		p.Destroy()
		return nil, err
	}
	if err := p.LoadAesthetics(d, pipeline.AesDir(models)); err != nil {
		p.Destroy()
		return nil, err
	}
	if err := p.LoadSongEval(d, models, plan.MaxSeconds); err != nil {
		p.Destroy()
		return nil, err
	}
	return p, nil
}

func request(s song) *plan.Request {
	return &plan.Request{Caption: s.Caption, Lyrics: s.Lyrics, Duration: s.Duration, Language: s.Lang}
}

// run renders one song's arms on p: on base, the plan first (the b32 arm).
func run(p *pipeline.Pipeline, out string, s song, dit string, rest []arm, seed uint64) error {
	r := request(s)
	planPath := filepath.Join(out, s.ID+".plan.json")
	var from *pipeline.Result
	if sp, err := readJSON[savedPlan](planPath); err == nil {
		from = &pipeline.Result{Meta: sp.Meta, CoT: sp.CoT, Codes: sp.Codes, DiT: sp.DiT}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if from == nil {
		if dit != "base" {
			return fmt.Errorf("no plan in %s: run the base pass first", planPath)
		}
		res, err := p.Generate(r, pipeline.Options{Seed: seed, Think: true, LMSeed: seed, Score: true})
		if err != nil {
			return err
		}
		if err := writeJSON(planPath, savedPlan{Meta: res.Meta, CoT: res.CoT, Codes: res.Codes, DiT: res.DiT}); err != nil {
			return err
		}
		if err := save(out, s, arms["b32"], res); err != nil {
			return err
		}
		from = res
	}
	for _, a := range rest {
		if _, err := os.Stat(filepath.Join(out, s.ID+"-"+a.name+".json")); err == nil {
			continue
		}
		o := pipeline.Options{Seed: seed, Score: true}
		if !a.turbo {
			o.Steps = a.steps
			g := plan.DefaultGuidance
			g.IntervalStart = a.cfgStart
			o.Guidance = &g
		}
		res, err := p.Render(r, from, o)
		if err != nil {
			return err
		}
		if err := save(out, s, a, res); err != nil {
			return err
		}
	}
	return nil
}

// save writes a render's mp3 and record.
func save(out string, s song, a arm, res *pipeline.Result) error {
	stem := filepath.Join(out, s.ID+"-"+a.name)
	if err := pipeline.Write(stem+".mp3", res.Audio); err != nil {
		return err
	}
	rec := record{Song: s.ID, Arm: a.name, Seconds: res.Seconds, DiT: res.Timings.DiT.Seconds(),
		Forwards: len(res.Timings.Steps), Metrics: metrics(res.Quality)}
	log.Printf("%s %-7s %5.1f s of audio, DiT %6.2f s: %s", s.ID, a.name, res.Seconds, rec.DiT, line(rec.Metrics))
	return writeJSON(stem+".json", rec)
}

// metricNames are the components the summary reports, in order.
var metricNames = []string{"PQ", "PC", "CE", "CU", "coherence", "musicality", "memorability", "clarity", "naturalness",
	"render_reward", "render_caption_pmi", "lyric_dit", "loop_share"}

func metrics(q *pipeline.Quality) map[string]float64 {
	m := map[string]float64{}
	if q == nil {
		return m
	}
	if a := q.Aesthetics; a != nil {
		m["PQ"], m["PC"], m["CE"], m["CU"] = a.PQ, a.PC, a.CE, a.CU
	}
	if s := q.Song; s != nil {
		m["coherence"], m["musicality"], m["memorability"] = s.Coherence, s.Musicality, s.Memorability
		m["clarity"], m["naturalness"] = s.Clarity, s.Naturalness
	}
	if r := q.Render; r != nil {
		m["render_reward"] = r.Reward
		if v, ok := r.PMI["caption"]; ok {
			m["render_caption_pmi"] = v
		}
	}
	if l := q.Lyrics; l != nil {
		m["lyric_dit"] = l.DiT.Score
	}
	if l := q.Loops; l != nil {
		m["loop_share"] = l.LoopShare
	}
	return m
}

func line(m map[string]float64) string {
	var parts []string
	for _, k := range metricNames {
		if v, ok := m[k]; ok {
			parts = append(parts, fmt.Sprintf("%s %.3f", k, v))
		}
	}
	return strings.Join(parts, ", ")
}

// summarise prints, per arm over the songs every arm rendered, the mean of
// each component, the mean paired difference from b32 (and on how many
// songs the arm scored at least b32's), and the DiT's seconds per second of
// song.
func summarise(out string, songs []song, armList string) {
	names := strings.Split(armList, ",")
	if !contains(names, "b32") {
		names = append([]string{"b32"}, names...)
	}
	recs := map[string]map[string]record{} // arm → song → record
	for _, a := range names {
		recs[a] = map[string]record{}
		for _, s := range songs {
			if r, err := readJSON[record](filepath.Join(out, s.ID+"-"+a+".json")); err == nil {
				recs[a][s.ID] = r
			}
		}
	}
	var common []string
	for _, s := range songs {
		all := true
		for _, a := range names {
			if _, ok := recs[a][s.ID]; !ok {
				all = false
			}
		}
		if all {
			common = append(common, s.ID)
		}
	}
	sort.Strings(common)
	fmt.Printf("\n%d songs rendered by every arm (%s)\n\n", len(common), strings.Join(names, ", "))
	fmt.Printf("%-20s", "")
	for _, a := range names {
		fmt.Printf(" %18s", a)
	}
	fmt.Println()
	for _, k := range append(metricNames, "dit_per_second") {
		fmt.Printf("%-20s", k)
		for _, a := range names {
			var sum, diff float64
			var n, ge int
			for _, id := range common {
				v, w, ok := value(recs[a][id], k), value(recs["b32"][id], k), true
				if math.IsNaN(v) || math.IsNaN(w) {
					ok = false
				}
				if !ok {
					continue
				}
				sum += v
				diff += v - w
				n++
				if v >= w {
					ge++
				}
			}
			if n == 0 {
				fmt.Printf(" %18s", "-")
				continue
			}
			if a == "b32" {
				fmt.Printf(" %18.3f", sum/float64(n))
			} else {
				fmt.Printf(" %8.3f %+6.3f %2d", sum/float64(n), diff/float64(n), ge)
			}
		}
		fmt.Println()
	}
	fmt.Println("\n(each arm: the mean, the mean paired difference from b32, and the songs it scored ≥ b32 on)")
}

func value(r record, k string) float64 {
	if k == "dit_per_second" {
		return r.DiT / r.Seconds
	}
	if v, ok := r.Metrics[k]; ok {
		return v
	}
	return math.NaN()
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

func readJSON[T any](path string) (T, error) {
	var v T
	raw, err := os.ReadFile(path)
	if err != nil {
		return v, err
	}
	return v, json.Unmarshal(raw, &v)
}

func writeJSON(path string, v any) error {
	raw, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

