package kokoro

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Blend is a voice: one pack's name, or several with the weights to mix them
// by.
//
// **The comma form is not ours.** hexgrad's own `KPipeline.load_voice` splits
// a voice on commas and returns `torch.mean(torch.stack(packs), dim=0)`, so
// "af_bella,af_sky" already means the equal mix everywhere else kokoro runs,
// and meaning anything else here would make the same request a different
// voice on this server. What is ours is the weight: "af_bella:3,af_sky:1" is
// an extension upstream has no spelling for.
//
// Weights are normalised by their sum, so 3 and 1 is the same mix as 0.75 and
// 0.25 and a client never has to make them add up. Either every component
// carries one or none does, because "af_bella:0.7,af_sky" has two readings --
// 0.3 for the rest, or 1 before normalising -- and a mix that silently picked
// one would still sound like a voice.
type Blend struct {
	Spec    string // as the caller wrote it, for the error messages
	Names   []string
	Weights []float32 // normalised: they sum to 1
}

// Single reports the one voice name when this is not a mixture, which is the
// path that has to stay exactly what it was: a single name indexes the pack
// in place and copies nothing.
func (b *Blend) Single() (string, bool) {
	if len(b.Names) == 1 {
		return b.Names[0], true
	}
	return "", false
}

// String is the canonical spelling of the blend, which is what a log line
// wants: the names it resolved to and the weights after normalisation.
func (b *Blend) String() string {
	if name, ok := b.Single(); ok {
		return name
	}
	parts := make([]string, len(b.Names))
	for i, n := range b.Names {
		parts[i] = fmt.Sprintf("%s:%.4g", n, b.Weights[i])
	}
	return strings.Join(parts, ",")
}

// ParseBlend reads a voice specification.
//
// It validates the spelling only: whether the names exist is the model's
// business, because the error for an unknown one wants to list what the
// checkpoint actually holds.
func ParseBlend(spec string) (*Blend, error) {
	fields := strings.Split(spec, ",")
	b := &Blend{
		Spec:    spec,
		Names:   make([]string, 0, len(fields)),
		Weights: make([]float32, 0, len(fields)),
	}
	weighted := 0
	var total float64
	for _, f := range fields {
		name, wtext, hasWeight := strings.Cut(f, ":")
		name = strings.TrimSpace(name)
		if name == "" {
			return nil, fmt.Errorf("kokoro: %q has an empty voice name in it", spec)
		}
		w := 1.0
		if hasWeight {
			weighted++
			var err error
			if w, err = strconv.ParseFloat(strings.TrimSpace(wtext), 64); err != nil {
				return nil, fmt.Errorf("kokoro: the weight on %q in %q is not a number", name, spec)
			}
			if w < 0 {
				return nil, fmt.Errorf("kokoro: the weight on %q in %q is negative", name, spec)
			}
		}
		total += w
		b.Names = append(b.Names, name)
		b.Weights = append(b.Weights, float32(w))
	}
	if weighted != 0 && weighted != len(b.Names) {
		return nil, fmt.Errorf("kokoro: %q weights %d of its %d voices; "+
			"weight all of them or none, since an unweighted component in a weighted mix "+
			"could mean either the remainder or one share of it", spec, weighted, len(b.Names))
	}
	if total == 0 {
		return nil, fmt.Errorf("kokoro: every weight in %q is zero", spec)
	}
	for i := range b.Weights {
		b.Weights[i] = float32(float64(b.Weights[i]) / total)
	}
	return b, nil
}

// CheckVoice reports whether a specification names a voice this checkpoint can
// speak, without synthesising anything. cmd/serve validates its -voice with it
// at startup, so a typo is a refusal to start rather than a 400 on the first
// request.
func (m *Model) CheckVoice(spec string) error {
	b, err := ParseBlend(spec)
	if err != nil {
		return err
	}
	_, err = m.blendRow(b, 1)
	return err
}

// blendRow resolves a blend to one style vector: row `row` of every named
// pack, weighted and summed.
//
// **The mix is per row, and that is the same number as upstream's.** Mixing
// whole packs and then indexing is what `load_voice` does; a weighted mean is
// linear, so the row of the mean is the mean of the rows, and this touches
// 256 numbers rather than 130560. The accumulation is float64 because it
// costs nothing at this size and a mean of 54 packs should not depend on the
// order they were named in.
func (m *Model) blendRow(b *Blend, row int) ([]float32, error) {
	var acc []float64
	for i, name := range b.Names {
		v, ok := m.Voices[name]
		if !ok {
			return nil, m.unknownVoice(name, b)
		}
		if acc == nil {
			acc = make([]float64, m.voiceDim)
		}
		w := float64(b.Weights[i])
		src := v[row*m.voiceDim : (row+1)*m.voiceDim]
		for j, x := range src {
			acc[j] += w * float64(x)
		}
	}
	out := make([]float32, len(acc))
	for j, x := range acc {
		out[j] = float32(x)
	}
	return out, nil
}

// UnknownVoiceError is a name that this checkpoint does not hold.
//
// It is a type rather than a string because the two callers want different
// messages out of the same fact: `cmd/tts` wants the 54 names, since a
// terminal is where you would look them up, and the HTTP layer wants the
// count and a pointer at `GET /v1/models`, since a 400 carrying 54 names is
// the same list every client already has an endpoint for.
type UnknownVoiceError struct {
	Name  string   // the component that was not found
	Blend string   // the whole specification, when it named more than one
	Have  []string // every voice in the checkpoint, sorted
}

func (e *UnknownVoiceError) Error() string {
	where := ""
	if e.Blend != "" {
		where = fmt.Sprintf(", named in the blend %q", e.Blend)
	}
	return fmt.Sprintf("kokoro: no voice %q%s (have %s)",
		e.Name, where, strings.Join(e.Have, ", "))
}

// unknownVoice names the component that was not found rather than the whole
// specification, which is the difference between "no voice
// af_alloy,af_bella,af_heart" and knowing which third of it to fix.
func (m *Model) unknownVoice(name string, b *Blend) error {
	e := &UnknownVoiceError{Name: name, Have: make([]string, 0, len(m.Voices))}
	for n := range m.Voices {
		e.Have = append(e.Have, n)
	}
	sort.Strings(e.Have)
	if _, single := b.Single(); !single {
		// The caller's spelling, not the canonical one: an error is
		// something they will look for in their own code, and
		// "af_bella:0.3333,..." is not what they wrote.
		e.Blend = b.Spec
	}
	return e
}
