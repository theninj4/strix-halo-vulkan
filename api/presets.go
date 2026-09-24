package api

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Presets: virtual models over the one set of weights.
//
// llama-server's `--models-preset` file is an INI of sections, one a model,
// and this server reads the same file. A section here carries no weights of
// its own: it is a name a request's `model` field can ask for, and a set of
// sampling and template settings that request then gets. What the file sets
// wins over the request, field by field -- a client that sends
// `reasoning_effort: "medium"` or `enable_thinking: true` to "chatting" still
// gets no thinking, since choosing the model is choosing its settings -- and
// what the file leaves out is still the request's to set.
//
// A key this server does not know is refused when the file is loaded rather
// than ignored, for the reason every other refusal in this package is there:
// a setting that is read and does nothing is a setting whoever wrote it
// believes is in force.

// Preset is one section of the file.
type Preset struct {
	Name            string
	Temperature     *float64
	TopP            *float64
	TopK            *int
	MinP            *float64
	RepeatPenalty   *float64
	PresencePenalty *float64
	// Kwargs is `chat-template-kwargs`, with `reasoning` folded in as
	// `enable_thinking`.
	Kwargs map[string]json.RawMessage
}

// LoadPresets reads a preset file.
func LoadPresets(path string) ([]Preset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	ps, err := ParsePresets(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return ps, nil
}

// ParsePresets reads the INI. llama-server spells its keys as its command
// line does, and the file this was written for mixes `top_p` and `top-p`, so
// both are taken; `;` and `#` start a comment line.
func ParsePresets(r io.Reader) ([]Preset, error) {
	var out []Preset
	var cur *Preset
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] == ';' || line[0] == '#' {
			continue
		}
		if line[0] == '[' {
			if !strings.HasSuffix(line, "]") {
				return nil, fmt.Errorf("line %d: unterminated section header", n)
			}
			name := strings.TrimSpace(line[1 : len(line)-1])
			if name == "" {
				return nil, fmt.Errorf("line %d: empty section name", n)
			}
			for _, p := range out {
				if p.Name == name {
					return nil, fmt.Errorf("line %d: section [%s] appears twice", n, name)
				}
			}
			out = append(out, Preset{Name: name})
			cur = &out[len(out)-1]
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("line %d: expected key = value", n)
		}
		if cur == nil {
			return nil, fmt.Errorf("line %d: %q is outside any section", n, strings.TrimSpace(key))
		}
		if err := cur.set(strings.TrimSpace(key), strings.TrimSpace(val)); err != nil {
			return nil, fmt.Errorf("line %d: [%s] %v", n, cur.Name, err)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// set applies one key.
func (p *Preset) set(key, val string) error {
	float := func(dst **float64) error {
		v, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return fmt.Errorf("%s: %v", key, err)
		}
		*dst = &v
		return nil
	}
	switch strings.ReplaceAll(strings.ToLower(key), "_", "-") {
	case "temperature", "temp":
		return float(&p.Temperature)
	case "top-p":
		return float(&p.TopP)
	case "min-p":
		return float(&p.MinP)
	case "repeat-penalty", "repetition-penalty":
		return float(&p.RepeatPenalty)
	case "presence-penalty":
		return float(&p.PresencePenalty)
	case "top-k":
		v, err := strconv.Atoi(val)
		if err != nil {
			return fmt.Errorf("%s: %v", key, err)
		}
		p.TopK = &v
		return nil
	case "chat-template-kwargs":
		var kw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(val), &kw); err != nil {
			return fmt.Errorf("%s: %v", key, err)
		}
		for k, v := range kw {
			if err := p.kwarg(k, v); err != nil {
				return err
			}
		}
		// The same check a request's kwargs get, so a typo fails here.
		_, err := (&CompletionRequest{ChatTemplateKwargs: p.Kwargs}).Thinking()
		return err
	case "reasoning":
		// llama-server's `--reasoning on|off|auto`: whether the template
		// opens a think block, which here is `enable_thinking`.
		switch strings.ToLower(val) {
		case "on":
			return p.kwarg("enable_thinking", json.RawMessage("true"))
		case "off":
			return p.kwarg("enable_thinking", json.RawMessage("false"))
		case "auto":
			return nil
		}
		return fmt.Errorf("reasoning is %q; it takes on, off or auto", val)
	}
	return fmt.Errorf("%q is not a setting this server applies per model; it takes temperature, top-p, top-k, "+
		"min-p, repeat-penalty, presence-penalty, chat-template-kwargs and reasoning", key)
}

// kwarg sets one template variable, refusing one set twice to two different
// values -- `reasoning = off` beside `enable_thinking: true` is a file that
// does not say what it wants.
func (p *Preset) kwarg(k string, v json.RawMessage) error {
	if p.Kwargs == nil {
		p.Kwargs = map[string]json.RawMessage{}
	}
	if old, ok := p.Kwargs[k]; ok && string(old) != string(v) {
		return fmt.Errorf("%s is set to both %s and %s", k, old, v)
	}
	p.Kwargs[k] = v
	return nil
}

// Apply overrides the request with every value the preset sets; what the
// preset leaves out stays the request's own.
func (p *Preset) Apply(req *CompletionRequest) {
	set := func(dst **float64, src *float64) {
		if src != nil {
			v := *src
			*dst = &v
		}
	}
	set(&req.Temperature, p.Temperature)
	set(&req.TopP, p.TopP)
	set(&req.MinP, p.MinP)
	set(&req.RepeatPenalty, p.RepeatPenalty)
	set(&req.PresencePenalty, p.PresencePenalty)
	if p.TopK != nil {
		v := *p.TopK
		req.TopK = &v
	}
	// A preset that says whether or how hard to think decides it: the
	// top-level `reasoning_effort` outranks the kwargs in Thinking, and
	// clients send it on every request whatever model they name.
	_, on := p.Kwargs["enable_thinking"]
	_, effort := p.Kwargs["reasoning_effort"]
	if on || effort {
		req.ReasoningEffort = ""
	}
	for k, v := range p.Kwargs {
		if req.ChatTemplateKwargs == nil {
			req.ChatTemplateKwargs = map[string]json.RawMessage{}
		}
		req.ChatTemplateKwargs[k] = v
	}
}

// preset is the preset a request's model names, or nil.
func (s *Server) preset(model string) *Preset {
	for i := range s.Presets {
		if s.Presets[i].Name == model {
			return &s.Presets[i]
		}
	}
	return nil
}

// completionModel applies the preset the request names, if it names one, and
// returns the model id the response reports: the preset's name, since that
// is what decided how the answer was sampled, or the backend's own id.
func (s *Server) completionModel(req *CompletionRequest) string {
	if p := s.preset(req.Model); p != nil {
		p.Apply(req)
		return p.Name
	}
	return modelID(s.Completion, req.Model)
}
