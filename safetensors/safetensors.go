// Package safetensors reads safetensors checkpoints without copying the
// weights onto the Go heap.
//
// A safetensors file is a little-endian uint64 header length, a JSON header
// mapping tensor name to {dtype, shape, data_offsets}, and then the raw
// tensor bytes. Offsets in the header are relative to the end of the header,
// so a tensor's bytes are always one contiguous span of the file.
//
// Why mmap rather than Read: Z-Image-Turbo's transformer is 23 GB of F32
// across three shards and its text encoder another 7.5 GB. Reading those
// into []byte would fault in every page and hold them in the heap for the
// lifetime of the process, when all we do with most of them is convert to
// fp16 once on the way to a GPU buffer. Mapping them lets the kernel page in
// what a conversion pass touches and evict it again behind us.
//
// The GOALS.md models arrive in three dtypes between them — F32 (the
// Z-Image transformer and VAE), BF16 (its Qwen3 text encoder) and F16 — and
// the fast path on this part is fp16 (IDEAS.md: WMMA fp16 measures 55.5
// TFLOP/s, and WMMA int8 is the same rate rather than double), so every
// dtype here converts to fp16 on the way out.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"syscall"
)

// DType is a safetensors dtype string. The float types the GOALS.md models
// compute in are handled end to end; I64 is carried but not convertible,
// because parakeet-tdt-0.6b-v3 stores each BatchNorm's scalar
// `num_batches_tracked` as one — 24 of its 723 tensors — and inference never
// reads them. Rejecting the file over those would be wrong, and skipping them
// at open would make cmd/inspect lie about what is in it, so they load like
// any other tensor and the float accessors refuse them at use. Every other
// dtype is still rejected at open time rather than at first use, so a bad
// checkpoint fails loudly and early.
type DType string

const (
	F64  DType = "F64"
	F32  DType = "F32"
	F16  DType = "F16"
	BF16 DType = "BF16"
	I64  DType = "I64"
)

// IsFloat reports whether a dtype carries a value the F16 and F32 accessors
// can convert.
func (d DType) IsFloat() bool {
	switch d {
	case F64, F32, F16, BF16:
		return true
	}
	return false
}

// Size is the width of one element in bytes.
func (d DType) Size() int {
	switch d {
	case F64, I64:
		return 8
	case F32:
		return 4
	case F16, BF16:
		return 2
	}
	return 0
}

// Tensor is one entry of a checkpoint. Data aliases the mapped file, so it
// stays valid until the owning File or Set is closed and must not be
// retained past that.
type Tensor struct {
	Name  string
	DType DType
	Shape []int
	Data  []byte
}

// Elems is the number of elements in the tensor.
func (t *Tensor) Elems() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// header is the JSON object at the head of every safetensors file.
type headerEntry struct {
	DType   DType  `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// File is one mapped safetensors file.
type File struct {
	Path    string
	tensors map[string]*Tensor
	order   []string
	mapping []byte
}

// Open maps a single safetensors file and parses its header.
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
	if size < 8 {
		return nil, fmt.Errorf("safetensors: %s is too short to hold a header length", path)
	}

	var lenBuf [8]byte
	if _, err := f.ReadAt(lenBuf[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: %s: reading header length: %w", path, err)
	}
	hdrLen := int64(binary.LittleEndian.Uint64(lenBuf[:]))
	if hdrLen <= 0 || 8+hdrLen > size {
		return nil, fmt.Errorf("safetensors: %s: header length %d does not fit in a %d byte file", path, hdrLen, size)
	}

	hdrBuf := make([]byte, hdrLen)
	if _, err := f.ReadAt(hdrBuf, 8); err != nil {
		return nil, fmt.Errorf("safetensors: %s: reading header: %w", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hdrBuf, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: %s: parsing header: %w", path, err)
	}

	// The whole file is mapped, header included, so that a tensor's Data can
	// be a subslice at its absolute offset with no arithmetic at use sites.
	mapping, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %s: mmap: %w", path, err)
	}

	file := &File{Path: path, tensors: make(map[string]*Tensor, len(raw)), mapping: mapping}
	dataStart := 8 + hdrLen

	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			file.Close()
			return nil, fmt.Errorf("safetensors: %s: tensor %q: %w", path, name, err)
		}
		if e.DType.Size() == 0 {
			file.Close()
			return nil, fmt.Errorf("safetensors: %s: tensor %q has unsupported dtype %q", path, name, e.DType)
		}
		lo, hi := dataStart+int64(e.Offsets[0]), dataStart+int64(e.Offsets[1])
		if lo < dataStart || hi > size || lo > hi {
			file.Close()
			return nil, fmt.Errorf("safetensors: %s: tensor %q offsets [%d,%d) out of range", path, name, e.Offsets[0], e.Offsets[1])
		}
		t := &Tensor{Name: name, DType: e.DType, Shape: e.Shape, Data: mapping[lo:hi]}
		if want := t.Elems() * e.DType.Size(); want != len(t.Data) {
			file.Close()
			return nil, fmt.Errorf("safetensors: %s: tensor %q: shape %v at %s needs %d bytes, header gives %d",
				path, name, e.Shape, e.DType, want, len(t.Data))
		}
		file.tensors[name] = t
		file.order = append(file.order, name)
	}
	return file, nil
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

// Set is a checkpoint that may be sharded over several files, as named by a
// model.safetensors.index.json "weight_map".
type Set struct {
	files   []*File
	tensors map[string]*Tensor
	order   []string
}

// OpenSet maps every shard of a checkpoint. If the directory holds an index
// (model.safetensors.index.json or diffusion_pytorch_model.safetensors.index.json)
// the shards it names are used; otherwise every *.safetensors file in the
// directory is mapped, which covers the single-file case such as the VAE.
func OpenSet(dir string) (*Set, error) {
	paths, err := shardPaths(dir)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("safetensors: no .safetensors files under %s", dir)
	}

	s := &Set{tensors: make(map[string]*Tensor)}
	for _, p := range paths {
		f, err := Open(p)
		if err != nil {
			s.Close()
			return nil, err
		}
		s.files = append(s.files, f)
		for _, name := range f.order {
			if _, dup := s.tensors[name]; dup {
				s.Close()
				return nil, fmt.Errorf("safetensors: tensor %q appears in more than one shard under %s", name, dir)
			}
			s.tensors[name] = f.tensors[name]
			s.order = append(s.order, name)
		}
	}
	return s, nil
}

// shardPaths resolves which files make up a checkpoint directory.
func shardPaths(dir string) ([]string, error) {
	indexes, err := filepath.Glob(filepath.Join(dir, "*.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	if len(indexes) > 1 {
		return nil, fmt.Errorf("safetensors: %s holds %d index files, expected one", dir, len(indexes))
	}
	if len(indexes) == 1 {
		buf, err := os.ReadFile(indexes[0])
		if err != nil {
			return nil, err
		}
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(buf, &idx); err != nil {
			return nil, fmt.Errorf("safetensors: parsing %s: %w", indexes[0], err)
		}
		seen := make(map[string]bool)
		var paths []string
		for _, shard := range idx.WeightMap {
			if !seen[shard] {
				seen[shard] = true
				paths = append(paths, filepath.Join(dir, shard))
			}
		}
		sortStrings(paths)
		return paths, nil
	}
	paths, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	sortStrings(paths)
	return paths, nil
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// Close unmaps every shard.
func (s *Set) Close() error {
	var first error
	for _, f := range s.files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
	}
	s.files = nil
	return first
}

// Get returns a tensor by name.
func (s *Set) Get(name string) (*Tensor, error) {
	t, ok := s.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor named %q", name)
	}
	return t, nil
}

// Has reports whether the checkpoint holds a tensor.
func (s *Set) Has(name string) bool { _, ok := s.tensors[name]; return ok }

// Names returns every tensor name, in the order the shards were read.
func (s *Set) Names() []string { return s.order }

// Len is the number of tensors.
func (s *Set) Len() int { return len(s.tensors) }

// Params is the total element count over every tensor.
func (s *Set) Params() int64 {
	var n int64
	for _, t := range s.tensors {
		n += int64(t.Elems())
	}
	return n
}

// Bytes is the total size on disk of every tensor's data.
func (s *Set) Bytes() int64 {
	var n int64
	for _, t := range s.tensors {
		n += int64(len(t.Data))
	}
	return n
}

// F16 converts a tensor to fp16, appending into dst so a caller can stage a
// whole layer into one buffer. F16 input is copied through unchanged.
//
// F32 and BF16 both truncate toward the nearest representable fp16 via
// float32 rounding; BF16 is an exact widening first (it is the top 16 bits
// of an F32), so the only lossy step for either is the final F32 -> F16.
func (t *Tensor) F16(dst []uint16) ([]uint16, error) {
	n := t.Elems()
	switch t.DType {
	case F16:
		for i := 0; i < n; i++ {
			dst = append(dst, binary.LittleEndian.Uint16(t.Data[i*2:]))
		}
	case BF16:
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(t.Data[i*2:])) << 16
			dst = append(dst, F32ToF16(math.Float32frombits(bits)))
		}
	case F32:
		for i := 0; i < n; i++ {
			dst = append(dst, F32ToF16(math.Float32frombits(binary.LittleEndian.Uint32(t.Data[i*4:]))))
		}
	case F64:
		for i := 0; i < n; i++ {
			dst = append(dst, F32ToF16(float32(math.Float64frombits(binary.LittleEndian.Uint64(t.Data[i*8:])))))
		}
	default:
		return nil, fmt.Errorf("safetensors: tensor %q: cannot convert dtype %q to fp16", t.Name, t.DType)
	}
	return dst, nil
}

// F32 converts a tensor to float32, appending into dst. This is the path for
// the tensors that stay in full precision — the norm weights, the adaLN
// modulation biases and anything feeding a reduction — where fp16's 10-bit
// mantissa is not enough.
func (t *Tensor) F32(dst []float32) ([]float32, error) {
	n := t.Elems()
	switch t.DType {
	case F32:
		for i := 0; i < n; i++ {
			dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(t.Data[i*4:])))
		}
	case BF16:
		for i := 0; i < n; i++ {
			dst = append(dst, math.Float32frombits(uint32(binary.LittleEndian.Uint16(t.Data[i*2:]))<<16))
		}
	case F16:
		for i := 0; i < n; i++ {
			dst = append(dst, F16ToF32(binary.LittleEndian.Uint16(t.Data[i*2:])))
		}
	case F64:
		for i := 0; i < n; i++ {
			dst = append(dst, float32(math.Float64frombits(binary.LittleEndian.Uint64(t.Data[i*8:]))))
		}
	default:
		return nil, fmt.Errorf("safetensors: tensor %q: cannot convert dtype %q to fp32", t.Name, t.DType)
	}
	return dst, nil
}

// F32ToF16 rounds a float32 to IEEE binary16, half-to-even, with overflow
// going to infinity and subnormals preserved.
//
// Exported because the narrowing is not only a loading concern: zimage/dit
// stages the DiT's weights into the device's fp16 arena in a layout the
// matrix cores want, which is a transpose or a retiling of what is on disk
// rather than a straight copy, so it cannot go through Tensor.F16.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127
	mant := b & 0x7fffff

	switch {
	case (b>>23)&0xff == 0xff: // Inf or NaN
		if mant != 0 {
			return sign | 0x7e00 // quiet NaN
		}
		return sign | 0x7c00
	case exp > 15:
		return sign | 0x7c00 // overflow to Inf
	case exp >= -14: // normal
		h := sign | uint16((exp+15)<<10) | uint16(mant>>13)
		if mant&0x1000 != 0 && (mant&0x0fff != 0 || h&1 != 0) {
			h++ // round half to even; carries into the exponent correctly
		}
		return h
	case exp >= -25: // subnormal, or an underflow that still rounds up to one
		// exp == -25 is below the smallest subnormal (2^-24) but at least
		// half of it, so it rounds to 2^-24 unless it is exactly the tie,
		// which goes to even i.e. zero. Anything smaller can never round up
		// and is caught by the default arm.
		// value = 1.mant x 2^exp, and an fp16 subnormal holds m x 2^-24,
		// so m = (2^23 | mant) >> (-exp - 1). Round half to even on the
		// bits that shift out; m carrying to 0x400 lands on the smallest
		// normal, which is the correct result rather than an overflow.
		shift := uint(-exp - 1)
		full := mant | 0x800000
		m := full >> shift
		rem := full & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && m&1 == 1) {
			m++
		}
		return sign | uint16(m)
	default:
		return sign // underflow to zero
	}
}

// F16ToF32 widens IEEE binary16 to float32. Exported alongside F32ToF16 for
// the same reason: an fp16 arena written by a Go-side packer is read back by
// Go-side validation, and the pair is already checked here against all 65,536
// bit patterns.
func F16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: the value is mant x 2^-24 exactly, and both factors
		// are exact in float32, so this needs no renormalisation loop.
		f := float32(mant) * (1.0 / 16777216.0)
		if sign != 0 {
			f = -f
		}
		return f
	case exp == 0x1f:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}
