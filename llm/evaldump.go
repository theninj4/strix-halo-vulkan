// Package llm is the qwen3.8-flash-next vertical: the model LLM.md builds,
// CPU reference first and then on the GPU, the same order every other
// vertical in this repo took.
//
// The oracle here is llama.cpp rather than a Python dump (LLM.md D5). Its own
// graph callback writes every intermediate tensor of a real forward pass to
// `reference/out/llm/` via `reference/eval_dump.c`, and the tests in this
// package read those files back and compare value for value. So "correct"
// means "the reference implementation's own numbers", not "a second reading
// of the paper".
package llm

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// A Dump is one tensor recorded by reference/eval_dump.c: the reference
// implementation's value for a named node of a real forward pass.
//
// Shape is ggml's, so NE[0] is the fastest-varying axis. Values is the
// payload converted to float32 and laid out in that same order, which means
// a ggml [features, tokens] tensor reads back as tokens-major rows of
// features — the layout the rest of this package uses.
type Dump struct {
	Name string
	Op   string
	Seq  int // graph order, from the filename
	NE   [4]int64
	NB   [4]uint64
	Vals []float32
	// Ints is the payload of an integer tensor, which is what the QSA
	// indexer's top-k selection is. Vals carries the same values widened, so
	// a shape check does not have to know which kind it is.
	Ints  []int32
	GType uint32
}

// Rows is the product of every axis but the first: how many NE[0]-long rows
// Values holds.
func (d *Dump) Rows() int {
	n := int64(1)
	for _, v := range d.NE[1:] {
		if v > 0 {
			n *= v
		}
	}
	return int(n)
}

// Row returns row i, of NE[0] values.
func (d *Dump) Row(i int) []float32 {
	w := int(d.NE[0])
	return d.Vals[i*w : (i+1)*w]
}

// ReadDump parses one .bin written by reference/eval_dump.c.
func ReadDump(path string) (*Dump, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 4+4+64+8 || string(b[:4]) != "EVDP" {
		return nil, fmt.Errorf("%s: not an eval_dump file", path)
	}
	le := binary.LittleEndian
	d := &Dump{GType: le.Uint32(b[4:])}
	p := 8
	for i := range d.NE {
		d.NE[i] = int64(le.Uint64(b[p:]))
		p += 8
	}
	for i := range d.NB {
		d.NB[i] = le.Uint64(b[p:])
		p += 8
	}
	nbytes := int(le.Uint64(b[p:]))
	p += 8
	rd := func() (string, error) {
		if p+4 > len(b) {
			return "", fmt.Errorf("%s: truncated", path)
		}
		n := int(le.Uint32(b[p:]))
		p += 4
		if p+n > len(b) {
			return "", fmt.Errorf("%s: truncated", path)
		}
		s := string(b[p : p+n])
		p += n
		return s, nil
	}
	if d.Name, err = rd(); err != nil {
		return nil, err
	}
	if d.Op, err = rd(); err != nil {
		return nil, err
	}
	if p+nbytes > len(b) {
		return nil, fmt.Errorf("%s: payload %d bytes, file has %d", path, nbytes, len(b)-p)
	}
	payload := b[p : p+nbytes]

	n := int(d.NE[0] * d.NE[1] * d.NE[2] * d.NE[3])
	d.Vals = make([]float32, n)
	switch d.GType {
	case 0: // f32
		if len(payload) < 4*n {
			return nil, fmt.Errorf("%s: f32 payload short", path)
		}
		for i := range d.Vals {
			d.Vals[i] = math.Float32frombits(le.Uint32(payload[4*i:]))
		}
	case 1: // f16
		for i := range d.Vals {
			d.Vals[i] = f16(le.Uint16(payload[2*i:]))
		}
	case 30: // bf16
		for i := range d.Vals {
			d.Vals[i] = math.Float32frombits(uint32(le.Uint16(payload[2*i:])) << 16)
		}
	case 26: // i32 — the QSA indexer's top-k is a list of cell indices
		if len(payload) < 4*n {
			return nil, fmt.Errorf("%s: i32 payload short", path)
		}
		d.Ints = make([]int32, n)
		for i := range d.Ints {
			d.Ints[i] = int32(le.Uint32(payload[4*i:]))
			d.Vals[i] = float32(d.Ints[i])
		}
	default:
		return nil, fmt.Errorf("%s: unhandled ggml type %d", path, d.GType)
	}
	if base := filepath.Base(path); len(base) > 5 && base[4] == '_' {
		fmt.Sscanf(base[:4], "%d", &d.Seq)
	}
	return d, nil
}

func f16(u uint16) float32 {
	sign := uint32(u&0x8000) << 16
	exp := int32(u>>10) & 0x1f
	man := uint32(u & 0x3ff)
	switch {
	case exp == 0:
		if man == 0 {
			return math.Float32frombits(sign)
		}
		e := int32(-1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		man &= 0x3ff
		return math.Float32frombits(sign | uint32(127-15+e+1)<<23 | man<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | man<<13)
	default:
		return math.Float32frombits(sign | uint32(exp-15+127)<<23 | man<<13)
	}
}

// A Trace is a whole dumped forward pass, indexed by name. A name can repeat
// — `build_hc_mix` runs twice a layer — so the entries are the writes in
// graph order and lookup takes an occurrence.
type Trace struct {
	Dir     string
	Entries []string // file paths in graph order
	byName  map[string][]string
}

// OpenTrace indexes a directory written by reference/eval_dump.c without
// reading the payloads.
func OpenTrace(dir string) (*Trace, error) {
	names, err := filepath.Glob(filepath.Join(dir, "*.bin"))
	if err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("%s: no eval_dump files (run reference/eval_dump.c)", dir)
	}
	sort.Strings(names)
	t := &Trace{Dir: dir, byName: map[string][]string{}}
	for _, p := range names {
		base := filepath.Base(p)
		if base == "tokens.bin" {
			continue
		}
		t.Entries = append(t.Entries, p)
		name := strings.TrimSuffix(base, ".bin")
		if len(name) > 5 && name[4] == '_' {
			name = name[5:]
		}
		t.byName[name] = append(t.byName[name], p)
	}
	return t, nil
}

// Get returns the nth write of a named tensor, 0 being the first.
func (t *Trace) Get(name string, nth int) (*Dump, error) {
	ps := t.byName[name]
	if nth >= len(ps) {
		return nil, fmt.Errorf("%s: %q written %d times, wanted #%d", t.Dir, name, len(ps), nth)
	}
	return ReadDump(ps[nth])
}

// Tokens returns the prompt's token ids, which are the other half of the
// fixture: without them the Go side cannot reproduce the same pass.
func (t *Trace) Tokens() ([]int32, string, error) {
	b, err := os.ReadFile(filepath.Join(t.Dir, "tokens.bin"))
	if err != nil {
		return nil, "", err
	}
	if len(b) < 8 || string(b[:4]) != "EVTK" {
		return nil, "", fmt.Errorf("%s/tokens.bin: bad magic", t.Dir)
	}
	le := binary.LittleEndian
	n := int(le.Uint32(b[4:]))
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(le.Uint32(b[8+4*i:]))
	}
	p := 8 + 4*n
	prompt := ""
	if p+4 <= len(b) {
		l := int(le.Uint32(b[p:]))
		if p+4+l <= len(b) {
			prompt = string(b[p+4 : p+4+l])
		}
	}
	return ids, prompt, nil
}
