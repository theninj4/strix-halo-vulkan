// Package gguf reads GGUF checkpoints without copying the weights onto the
// Go heap.
//
// A GGUF file is a magic, a version, two counts, a metadata table of typed
// key/value pairs, a table of tensor descriptors, padding to
// `general.alignment`, and then one contiguous data section that every
// descriptor's offset is relative to. Unlike safetensors it carries the
// hyper-parameters as well as the weights, so the reader has to understand
// the value grammar and not only the tensor table.
//
// Why this exists alongside safetensors/: LLM.md's D1 has this project
// consuming `unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL` directly — 111.32 GB
// over four shards, quantised with an imatrix that would cost 360 GB of bf16
// and a calibration run to reproduce — and llama.cpp is then a bit-exact
// oracle for it (D5). Reading the container is the first half of that; the
// dequant paths for the five formats it actually uses (Q4_K, Q5_1, Q8_0,
// Q5_K, IQ4_NL) are the second.
//
// Why mmap, same as safetensors: a shard is 50 GB. The whole point of the
// format work in LLM.md is that weights are read once on the way into a
// device buffer, so they should never be resident on the Go heap.
//
// Shapes are in ggml order: Dims[0] is the fastest-varying axis, which for a
// 2-D weight is the row length. That is the transpose of the torch
// convention safetensors/ carries, and it is kept as the file states it
// rather than normalised, because every offset computed against a quant
// block is an offset along Dims[0].
package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"syscall"
)

// Magic is the four bytes every GGUF file starts with.
const Magic = "GGUF"

// DefaultAlignment is the data-section alignment assumed when a file carries
// no `general.alignment` key.
const DefaultAlignment = 32

// Type is a ggml tensor type — the quantisation format of one tensor. The
// numbering is ggml's own and is part of the file format, so the constants
// are written out rather than iota'd.
type Type uint32

const (
	F32     Type = 0
	F16     Type = 1
	Q4_0    Type = 2
	Q4_1    Type = 3
	Q5_0    Type = 6
	Q5_1    Type = 7
	Q8_0    Type = 8
	Q8_1    Type = 9
	Q2_K    Type = 10
	Q3_K    Type = 11
	Q4_K    Type = 12
	Q5_K    Type = 13
	Q6_K    Type = 14
	Q8_K    Type = 15
	IQ2_XXS Type = 16
	IQ2_XS  Type = 17
	IQ3_XXS Type = 18
	IQ1_S   Type = 19
	IQ4_NL  Type = 20
	IQ3_S   Type = 21
	IQ2_S   Type = 22
	IQ4_XS  Type = 23
	I8      Type = 24
	I16     Type = 25
	I32     Type = 26
	I64     Type = 27
	F64     Type = 28
	IQ1_M   Type = 29
	BF16    Type = 30
	TQ1_0   Type = 34
	TQ2_0   Type = 35
	MXFP4   Type = 39
)

// typeInfo is a type's name and its block geometry: how many elements one
// block holds and how many bytes it occupies. The numbers are ggml-common.h's
// and are mirrored by reference/gguf_inventory.py, which produced LLM.md's
// inventory tables — the two tables agreeing is what makes the Go reader
// checkable against the Python one.
type typeInfo struct {
	name   string
	elems  int
	nbytes int
}

var types = map[Type]typeInfo{
	F32:     {"F32", 1, 4},
	F16:     {"F16", 1, 2},
	BF16:    {"BF16", 1, 2},
	F64:     {"F64", 1, 8},
	I8:      {"I8", 1, 1},
	I16:     {"I16", 1, 2},
	I32:     {"I32", 1, 4},
	I64:     {"I64", 1, 8},
	Q4_0:    {"Q4_0", 32, 18},
	Q4_1:    {"Q4_1", 32, 20},
	Q5_0:    {"Q5_0", 32, 22},
	Q5_1:    {"Q5_1", 32, 24},
	Q8_0:    {"Q8_0", 32, 34},
	Q8_1:    {"Q8_1", 32, 36},
	IQ4_NL:  {"IQ4_NL", 32, 18},
	MXFP4:   {"MXFP4", 32, 17},
	Q2_K:    {"Q2_K", 256, 84},
	Q3_K:    {"Q3_K", 256, 110},
	Q4_K:    {"Q4_K", 256, 144},
	Q5_K:    {"Q5_K", 256, 176},
	Q6_K:    {"Q6_K", 256, 210},
	Q8_K:    {"Q8_K", 256, 292},
	IQ4_XS:  {"IQ4_XS", 256, 136},
	IQ3_S:   {"IQ3_S", 256, 110},
	IQ3_XXS: {"IQ3_XXS", 256, 98},
	IQ2_XXS: {"IQ2_XXS", 256, 66},
	IQ2_XS:  {"IQ2_XS", 256, 74},
	IQ2_S:   {"IQ2_S", 256, 82},
	IQ1_S:   {"IQ1_S", 256, 50},
	IQ1_M:   {"IQ1_M", 256, 56},
	TQ1_0:   {"TQ1_0", 256, 54},
	TQ2_0:   {"TQ2_0", 256, 66},
}

// String is the ggml name of the type, or `?<n>` for one this table does not
// know, so an unknown type in a checkpoint is reported rather than guessed at.
func (t Type) String() string {
	if info, ok := types[t]; ok {
		return info.name
	}
	return "?" + strconv.FormatUint(uint64(t), 10)
}

// Known reports whether the block geometry of the type is known, which is the
// precondition for computing any tensor's size.
func (t Type) Known() bool { _, ok := types[t]; return ok }

// BlockElems is how many elements one block of the type holds; 1 for the
// unquantised types.
func (t Type) BlockElems() int { return types[t].elems }

// BlockBytes is how many bytes one block of the type occupies.
func (t Type) BlockBytes() int { return types[t].nbytes }

// Quantised reports whether the type packs more than one element per block.
func (t Type) Quantised() bool { return types[t].elems > 1 }

// SizeOf is the on-disk size of n elements of the type. n must be a multiple
// of the block size, which ggml guarantees for Dims[0].
func (t Type) SizeOf(n int64) (int64, error) {
	info, ok := types[t]
	if !ok {
		return 0, fmt.Errorf("gguf: unknown ggml type %d", uint32(t))
	}
	if n%int64(info.elems) != 0 {
		return 0, fmt.Errorf("gguf: %d elements is not a whole number of %s blocks of %d", n, info.name, info.elems)
	}
	return n / int64(info.elems) * int64(info.nbytes), nil
}

// Tensor is one entry of the tensor table. Data aliases the mapped shard, so
// it stays valid until the owning File or Set is closed and must not be
// retained past that.
type Tensor struct {
	Name  string
	Type  Type
	Dims  []int64 // ggml order: Dims[0] is the fastest-varying axis
	Data  []byte
	Shard int // index into Set.Files, 0 for a single file
}

// Elems is the number of elements in the tensor.
func (t *Tensor) Elems() int64 {
	n := int64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// Rows is the number of rows — the product of every axis but the first. A
// quantised tensor is blocked along Dims[0] only, so a row is the unit every
// dequant path works in.
func (t *Tensor) Rows() int64 {
	n := int64(1)
	for _, d := range t.Dims[1:] {
		n *= d
	}
	return n
}

// RowBytes is the on-disk size of one row.
func (t *Tensor) RowBytes() int64 {
	n, err := t.Type.SizeOf(t.Dims[0])
	if err != nil {
		return 0
	}
	return n
}

// Row is the bytes of row i, which for a quantised type is a whole number of
// blocks.
func (t *Tensor) Row(i int64) []byte {
	rb := t.RowBytes()
	return t.Data[i*rb : (i+1)*rb]
}

// File is one mapped GGUF file: a whole checkpoint, or one shard of a split
// one.
type File struct {
	Path      string
	Version   uint32
	Alignment uint64
	KV        map[string]any
	kvOrder   []string
	Tensors   []*Tensor
	byName    map[string]*Tensor
	DataStart int64 // file offset of the data section
	mapping   []byte
}

// Open maps a GGUF file and parses its metadata and tensor table.
//
// The whole file is mapped, header included, so a tensor's Data can be a
// subslice at its absolute offset with no arithmetic at use sites — the same
// choice safetensors.Open makes, for the same reason.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if size < 24 {
		return nil, fmt.Errorf("gguf: %s is too short to hold a header", path)
	}
	mapping, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("gguf: %s: mmap: %w", path, err)
	}
	file := &File{Path: path, Alignment: DefaultAlignment, KV: map[string]any{}, mapping: mapping}

	if err := file.parse(); err != nil {
		file.Close()
		return nil, fmt.Errorf("gguf: %s: %w", path, err)
	}
	return file, nil
}

// parse walks the header of an already-mapped file.
func (f *File) parse() error {
	r := &reader{b: f.mapping}
	if string(f.mapping[:4]) != Magic {
		return fmt.Errorf("not a GGUF file (magic %q)", f.mapping[:4])
	}
	r.o = 4
	var err error
	if f.Version, err = r.u32(); err != nil {
		return err
	}
	if f.Version < 2 || f.Version > 3 {
		return fmt.Errorf("unsupported GGUF version %d", f.Version)
	}
	nTensor, err := r.u64()
	if err != nil {
		return err
	}
	nKV, err := r.u64()
	if err != nil {
		return err
	}
	for i := uint64(0); i < nKV; i++ {
		k, err := r.str()
		if err != nil {
			return fmt.Errorf("metadata key %d: %w", i, err)
		}
		vt, err := r.u32()
		if err != nil {
			return fmt.Errorf("metadata %q: %w", k, err)
		}
		v, err := r.value(valueType(vt))
		if err != nil {
			return fmt.Errorf("metadata %q: %w", k, err)
		}
		if _, dup := f.KV[k]; !dup {
			f.kvOrder = append(f.kvOrder, k)
		}
		f.KV[k] = v
	}
	if a, ok := f.Uint(KeyAlignment); ok {
		if a == 0 || a&(a-1) != 0 {
			return fmt.Errorf("general.alignment %d is not a power of two", a)
		}
		f.Alignment = a
	}

	// The tensor table is read before the data section's start is known,
	// because the table is what ends the header.
	type desc struct {
		name   string
		typ    Type
		dims   []int64
		offset uint64
	}
	descs := make([]desc, 0, nTensor)
	for i := uint64(0); i < nTensor; i++ {
		var d desc
		if d.name, err = r.str(); err != nil {
			return fmt.Errorf("tensor %d: %w", i, err)
		}
		nd, err := r.u32()
		if err != nil {
			return fmt.Errorf("tensor %q: %w", d.name, err)
		}
		if nd == 0 || nd > 4 {
			return fmt.Errorf("tensor %q has %d dimensions, ggml allows 1-4", d.name, nd)
		}
		d.dims = make([]int64, nd)
		for j := range d.dims {
			v, err := r.u64()
			if err != nil {
				return fmt.Errorf("tensor %q: %w", d.name, err)
			}
			d.dims[j] = int64(v)
		}
		t, err := r.u32()
		if err != nil {
			return fmt.Errorf("tensor %q: %w", d.name, err)
		}
		d.typ = Type(t)
		if d.offset, err = r.u64(); err != nil {
			return fmt.Errorf("tensor %q: %w", d.name, err)
		}
		descs = append(descs, d)
	}

	f.DataStart = int64((uint64(r.o) + f.Alignment - 1) &^ (f.Alignment - 1))
	f.byName = make(map[string]*Tensor, len(descs))
	for _, d := range descs {
		t := &Tensor{Name: d.name, Type: d.typ, Dims: d.dims}
		n := t.Elems()
		nbytes, err := d.typ.SizeOf(n)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", d.name, err)
		}
		lo := f.DataStart + int64(d.offset)
		hi := lo + nbytes
		if lo < f.DataStart || hi > int64(len(f.mapping)) {
			return fmt.Errorf("tensor %q spans [%d,%d), past the end of a %d byte file",
				d.name, lo, hi, len(f.mapping))
		}
		t.Data = f.mapping[lo:hi]
		if _, dup := f.byName[d.name]; dup {
			return fmt.Errorf("tensor %q appears twice", d.name)
		}
		f.byName[d.name] = t
		f.Tensors = append(f.Tensors, t)
	}
	return nil
}

// Close unmaps the file. Every Tensor.Data from it dangles afterwards.
func (f *File) Close() error {
	if f.mapping == nil {
		return nil
	}
	err := syscall.Munmap(f.mapping)
	f.mapping = nil
	return err
}

// Keys returns every metadata key in file order.
func (f *File) Keys() []string { return f.kvOrder }

// Get returns a tensor by name.
func (f *File) Get(name string) (*Tensor, error) {
	t, ok := f.byName[name]
	if !ok {
		return nil, fmt.Errorf("gguf: %s: no tensor named %q", f.Path, name)
	}
	return t, nil
}

// Has reports whether the file holds a tensor.
func (f *File) Has(name string) bool { _, ok := f.byName[name]; return ok }

// reader walks the little-endian header.
type reader struct {
	b []byte
	o int
}

func (r *reader) need(n int) error {
	if r.o+n > len(r.b) {
		return fmt.Errorf("truncated at offset %d, wanted %d more bytes of %d", r.o, n, len(r.b))
	}
	return nil
}

func (r *reader) u8() (uint8, error) {
	if err := r.need(1); err != nil {
		return 0, err
	}
	v := r.b[r.o]
	r.o++
	return v, nil
}

func (r *reader) u16() (uint16, error) {
	if err := r.need(2); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint16(r.b[r.o:])
	r.o += 2
	return v, nil
}

func (r *reader) u32() (uint32, error) {
	if err := r.need(4); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(r.b[r.o:])
	r.o += 4
	return v, nil
}

func (r *reader) u64() (uint64, error) {
	if err := r.need(8); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(r.b[r.o:])
	r.o += 8
	return v, nil
}

// str reads a GGUF string: a u64 length and that many bytes of UTF-8.
//
// The bytes are copied rather than aliased. Strings are the one part of the
// header that outlives the parse — the 248 320-entry tokenizer vocabulary is
// 1224 tensors' worth of metadata on its own — and a Go string aliasing the
// mapping would keep every use site tied to the file's lifetime for no gain.
func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > uint64(len(r.b)) {
		return "", fmt.Errorf("string length %d exceeds the file", n)
	}
	if err := r.need(int(n)); err != nil {
		return "", err
	}
	s := string(r.b[r.o : r.o+int(n)])
	r.o += int(n)
	return s, nil
}

// valueType is the tag on a metadata value.
type valueType uint32

const (
	vtUint8 valueType = iota
	vtInt8
	vtUint16
	vtInt16
	vtUint32
	vtInt32
	vtFloat32
	vtBool
	vtString
	vtArray
	vtUint64
	vtInt64
	vtFloat64
)

// value reads one metadata value.
//
// Scalars widen to uint64/int64/float64/string/bool, so that a caller asking
// for `qwen4exp.expert_count` does not have to know whether the writer chose
// u32 or u64 for it. Arrays keep their element type's widened form, one
// []uint64 / []int64 / []float64 / []string / []bool per array, rather than
// []any — the tokenizer's three big arrays are 248 320 entries each and an
// []any of those would be 248 320 allocations.
func (r *reader) value(t valueType) (any, error) {
	switch t {
	case vtUint8:
		v, err := r.u8()
		return uint64(v), err
	case vtInt8:
		v, err := r.u8()
		return int64(int8(v)), err
	case vtUint16:
		v, err := r.u16()
		return uint64(v), err
	case vtInt16:
		v, err := r.u16()
		return int64(int16(v)), err
	case vtUint32:
		v, err := r.u32()
		return uint64(v), err
	case vtInt32:
		v, err := r.u32()
		return int64(int32(v)), err
	case vtFloat32:
		v, err := r.u32()
		return float64(math.Float32frombits(v)), err
	case vtBool:
		v, err := r.u8()
		return v != 0, err
	case vtString:
		return r.str()
	case vtUint64:
		v, err := r.u64()
		return v, err
	case vtInt64:
		v, err := r.u64()
		return int64(v), err
	case vtFloat64:
		v, err := r.u64()
		return math.Float64frombits(v), err
	case vtArray:
		et, err := r.u32()
		if err != nil {
			return nil, err
		}
		n, err := r.u64()
		if err != nil {
			return nil, err
		}
		return r.array(valueType(et), n)
	}
	return nil, fmt.Errorf("unknown value type %d", uint32(t))
}

// array reads n values of one element type into a typed slice.
func (r *reader) array(et valueType, n uint64) (any, error) {
	if n > uint64(len(r.b)) {
		return nil, fmt.Errorf("array of %d elements exceeds the file", n)
	}
	switch et {
	case vtString:
		out := make([]string, n)
		for i := range out {
			v, err := r.str()
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case vtBool:
		out := make([]bool, n)
		for i := range out {
			v, err := r.value(et)
			if err != nil {
				return nil, err
			}
			out[i] = v.(bool)
		}
		return out, nil
	case vtFloat32, vtFloat64:
		out := make([]float64, n)
		for i := range out {
			v, err := r.value(et)
			if err != nil {
				return nil, err
			}
			out[i] = v.(float64)
		}
		return out, nil
	case vtInt8, vtInt16, vtInt32, vtInt64:
		out := make([]int64, n)
		for i := range out {
			v, err := r.value(et)
			if err != nil {
				return nil, err
			}
			out[i] = v.(int64)
		}
		return out, nil
	case vtUint8, vtUint16, vtUint32, vtUint64:
		out := make([]uint64, n)
		for i := range out {
			v, err := r.value(et)
			if err != nil {
				return nil, err
			}
			out[i] = v.(uint64)
		}
		return out, nil
	case vtArray:
		return nil, fmt.Errorf("nested arrays are not in the GGUF grammar")
	}
	return nil, fmt.Errorf("unknown array element type %d", uint32(et))
}

// The metadata keys this project reads by name. The split keys are the ones
// that make a four-shard checkpoint one model rather than four.
const (
	KeyAlignment    = "general.alignment"
	KeyArchitecture = "general.architecture"
	KeySplitNo      = "split.no"
	KeySplitCount   = "split.count"
	KeySplitTensors = "split.tensors.count"
)

// Uint reads a metadata value as a uint64, accepting any of GGUF's unsigned
// or signed integer widths.
func (f *File) Uint(key string) (uint64, bool) {
	switch v := f.KV[key].(type) {
	case uint64:
		return v, true
	case int64:
		if v >= 0 {
			return uint64(v), true
		}
	}
	return 0, false
}

// Int reads a metadata value as an int64.
func (f *File) Int(key string) (int64, bool) {
	switch v := f.KV[key].(type) {
	case int64:
		return v, true
	case uint64:
		return int64(v), true
	}
	return 0, false
}

// Float reads a metadata value as a float64.
func (f *File) Float(key string) (float64, bool) {
	switch v := f.KV[key].(type) {
	case float64:
		return v, true
	case uint64:
		return float64(v), true
	case int64:
		return float64(v), true
	}
	return 0, false
}

// Str reads a metadata value as a string.
func (f *File) Str(key string) (string, bool) {
	v, ok := f.KV[key].(string)
	return v, ok
}

// Bool reads a metadata value as a bool.
func (f *File) Bool(key string) (bool, bool) {
	v, ok := f.KV[key].(bool)
	return v, ok
}

// Strings reads a metadata value as a string array — the tokenizer's tokens,
// merges and token types.
func (f *File) Strings(key string) ([]string, bool) {
	v, ok := f.KV[key].([]string)
	return v, ok
}

// Uints reads a metadata value as an unsigned integer array.
func (f *File) Uints(key string) ([]uint64, bool) {
	v, ok := f.KV[key].([]uint64)
	return v, ok
}

// Ints reads a metadata value as a signed integer array.
func (f *File) Ints(key string) ([]int64, bool) {
	v, ok := f.KV[key].([]int64)
	return v, ok
}

// Floats reads a metadata value as a float array.
func (f *File) Floats(key string) ([]float64, bool) {
	v, ok := f.KV[key].([]float64)
	return v, ok
}

// Set is a checkpoint that may be split over several shards.
//
// llama.cpp's split convention: every shard is a complete GGUF with its own
// tensor table, shard 1 carries the model's metadata, and each shard carries
// `split.no`, `split.count` and `split.tensors.count`. Names follow
// `<base>-00001-of-00004.gguf`, which is how the siblings are found from any
// one of them.
type Set struct {
	Files   []*File
	Tensors []*Tensor
	byName  map[string]*Tensor
}

// splitName matches the shard suffix llama-gguf-split writes.
var splitName = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// OpenSet maps a checkpoint given any one of its shards, or a directory
// holding exactly one checkpoint.
//
// Passing shard 2 of 4 works and opens all four: the suffix names the count,
// so the set is resolvable from any member. That matters because the shard a
// caller has a path to is not always the first one.
func OpenSet(path string) (*Set, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if st.IsDir() {
		if path, err = soleCheckpoint(path); err != nil {
			return nil, err
		}
	}
	paths, err := shardPaths(path)
	if err != nil {
		return nil, err
	}

	s := &Set{byName: map[string]*Tensor{}}
	for i, p := range paths {
		f, err := Open(p)
		if err != nil {
			s.Close()
			return nil, err
		}
		s.Files = append(s.Files, f)
		for _, t := range f.Tensors {
			if _, dup := s.byName[t.Name]; dup {
				s.Close()
				return nil, fmt.Errorf("gguf: tensor %q appears in more than one shard of %s", t.Name, path)
			}
			t.Shard = i
			s.byName[t.Name] = t
			s.Tensors = append(s.Tensors, t)
		}
	}
	if err := s.checkSplit(); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

// checkSplit verifies the shards form one complete checkpoint: consecutive
// `split.no`, an agreed `split.count`, and the promised tensor total.
//
// This is cheap and it catches the failure mode that matters — a download
// that stopped after three of four shards still opens, and every tensor in
// it still parses, so nothing else would notice until a weight came back
// missing halfway through a model load.
func (s *Set) checkSplit() error {
	if len(s.Files) == 1 {
		if n, ok := s.Files[0].Uint(KeySplitCount); ok && n > 1 {
			return fmt.Errorf("gguf: %s is shard 1 of %d but no siblings were found", s.Files[0].Path, n)
		}
		return nil
	}
	for i, f := range s.Files {
		no, ok := f.Uint(KeySplitNo)
		if !ok {
			return fmt.Errorf("gguf: %s has no %s", f.Path, KeySplitNo)
		}
		if no != uint64(i) {
			return fmt.Errorf("gguf: %s is %s=%d, expected %d", f.Path, KeySplitNo, no, i)
		}
		if n, ok := f.Uint(KeySplitCount); ok && n != uint64(len(s.Files)) {
			return fmt.Errorf("gguf: %s says %s=%d but %d shards were opened", f.Path, KeySplitCount, n, len(s.Files))
		}
	}
	if want, ok := s.Files[0].Uint(KeySplitTensors); ok && want != uint64(len(s.Tensors)) {
		return fmt.Errorf("gguf: %s says %s=%d but %d tensors were read", s.Files[0].Path, KeySplitTensors, want, len(s.Tensors))
	}
	return nil
}

// soleCheckpoint picks the one checkpoint in a directory, which is either a
// single .gguf or the first shard of one split set.
func soleCheckpoint(dir string) (string, error) {
	all, err := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if err != nil {
		return "", err
	}
	var candidates []string
	for _, p := range all {
		m := splitName.FindStringSubmatch(filepath.Base(p))
		if m == nil || m[2] == "00001" {
			candidates = append(candidates, p)
		}
	}
	sort.Strings(candidates)
	switch len(candidates) {
	case 0:
		return "", fmt.Errorf("gguf: no .gguf checkpoint under %s", dir)
	case 1:
		return candidates[0], nil
	}
	return "", fmt.Errorf("gguf: %s holds %d checkpoints (%v), name one", dir, len(candidates), candidates)
}

// shardPaths expands one shard's path into every shard of its set.
func shardPaths(path string) ([]string, error) {
	m := splitName.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return []string{path}, nil
	}
	count, err := strconv.Atoi(m[3])
	if err != nil || count < 1 {
		return nil, fmt.Errorf("gguf: %s: unreadable shard count %q", path, m[3])
	}
	dir := filepath.Dir(path)
	paths := make([]string, count)
	for i := 0; i < count; i++ {
		paths[i] = filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", m[1], i+1, count))
		if _, err := os.Stat(paths[i]); err != nil {
			return nil, fmt.Errorf("gguf: %s: shard %d of %d is missing: %w", path, i+1, count, err)
		}
	}
	return paths, nil
}

// Close unmaps every shard.
func (s *Set) Close() error {
	var first error
	for _, f := range s.Files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
	}
	s.Files = nil
	return first
}

// KVFile is the shard that carries the model metadata — shard 1 by the split
// convention. Every KV accessor on Set reads it.
func (s *Set) KVFile() *File { return s.Files[0] }

// Get returns a tensor by name from whichever shard holds it.
func (s *Set) Get(name string) (*Tensor, error) {
	t, ok := s.byName[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor named %q", name)
	}
	return t, nil
}

// Has reports whether the checkpoint holds a tensor.
func (s *Set) Has(name string) bool { _, ok := s.byName[name]; return ok }

// Len is the number of tensors.
func (s *Set) Len() int { return len(s.Tensors) }

// Params is the total element count over every tensor.
func (s *Set) Params() int64 {
	var n int64
	for _, t := range s.Tensors {
		n += t.Elems()
	}
	return n
}

// Bytes is the total size of every tensor's data, which is the checkpoint's
// size less its headers.
func (s *Set) Bytes() int64 {
	var n int64
	for _, t := range s.Tensors {
		n += int64(len(t.Data))
	}
	return n
}

// Arch is the model architecture, `qwen4exp` for this project's target.
func (s *Set) Arch() string { v, _ := s.KVFile().Str(KeyArchitecture); return v }

// The KV accessors on Set forward to shard 1.
func (s *Set) Uint(k string) (uint64, bool)      { return s.KVFile().Uint(k) }
func (s *Set) Int(k string) (int64, bool)        { return s.KVFile().Int(k) }
func (s *Set) Float(k string) (float64, bool)    { return s.KVFile().Float(k) }
func (s *Set) Str(k string) (string, bool)       { return s.KVFile().Str(k) }
func (s *Set) Bool(k string) (bool, bool)        { return s.KVFile().Bool(k) }
func (s *Set) Strings(k string) ([]string, bool) { return s.KVFile().Strings(k) }
func (s *Set) Uints(k string) ([]uint64, bool)   { return s.KVFile().Uints(k) }
func (s *Set) Ints(k string) ([]int64, bool)     { return s.KVFile().Ints(k) }
func (s *Set) Floats(k string) ([]float64, bool) { return s.KVFile().Floats(k) }

// Keys returns shard 1's metadata keys in file order.
func (s *Set) Keys() []string { return s.KVFile().Keys() }
