package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The tests build GGUF files rather than checking one in: the target
// checkpoint is 111 GB and even one shard of it is 50, so the only way to
// have the container's edge cases — every value type, a split set, a bad
// offset — in a test is to write them.

// writer assembles a GGUF byte by byte, in the file's own grammar.
type writer struct {
	kv      bytes.Buffer
	nKV     uint64
	desc    bytes.Buffer
	nTensor uint64
	data    bytes.Buffer
	align   uint64
}

func newWriter() *writer { return &writer{align: DefaultAlignment} }

func (w *writer) str(b *bytes.Buffer, s string) {
	binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func (w *writer) put(key string, vt valueType, write func(*bytes.Buffer)) {
	w.str(&w.kv, key)
	binary.Write(&w.kv, binary.LittleEndian, uint32(vt))
	write(&w.kv)
	w.nKV++
}

func (w *writer) putU32(key string, v uint32) {
	w.put(key, vtUint32, func(b *bytes.Buffer) { binary.Write(b, binary.LittleEndian, v) })
}

func (w *writer) putU16(key string, v uint16) {
	w.put(key, vtUint16, func(b *bytes.Buffer) { binary.Write(b, binary.LittleEndian, v) })
}

func (w *writer) putI32(key string, v int32) {
	w.put(key, vtInt32, func(b *bytes.Buffer) { binary.Write(b, binary.LittleEndian, v) })
}

func (w *writer) putF32(key string, v float32) {
	w.put(key, vtFloat32, func(b *bytes.Buffer) { binary.Write(b, binary.LittleEndian, math.Float32bits(v)) })
}

func (w *writer) putBool(key string, v bool) {
	w.put(key, vtBool, func(b *bytes.Buffer) {
		var x byte
		if v {
			x = 1
		}
		b.WriteByte(x)
	})
}

func (w *writer) putStr(key, v string) {
	w.put(key, vtString, func(b *bytes.Buffer) { w.str(b, v) })
}

func (w *writer) putStrArray(key string, vs []string) {
	w.put(key, vtArray, func(b *bytes.Buffer) {
		binary.Write(b, binary.LittleEndian, uint32(vtString))
		binary.Write(b, binary.LittleEndian, uint64(len(vs)))
		for _, v := range vs {
			w.str(b, v)
		}
	})
}

func (w *writer) putI32Array(key string, vs []int32) {
	w.put(key, vtArray, func(b *bytes.Buffer) {
		binary.Write(b, binary.LittleEndian, uint32(vtInt32))
		binary.Write(b, binary.LittleEndian, uint64(len(vs)))
		for _, v := range vs {
			binary.Write(b, binary.LittleEndian, v)
		}
	})
}

// tensor appends a descriptor and its bytes, padding the data section so
// every tensor starts on an alignment boundary the way ggml writes them.
func (w *writer) tensor(name string, typ Type, dims []int64, fill byte) {
	for w.data.Len()%int(w.align) != 0 {
		w.data.WriteByte(0)
	}
	off := uint64(w.data.Len())
	n := int64(1)
	for _, d := range dims {
		n *= d
	}
	nbytes, err := typ.SizeOf(n)
	if err != nil {
		panic(err)
	}
	for i := int64(0); i < nbytes; i++ {
		w.data.WriteByte(fill)
	}
	w.str(&w.desc, name)
	binary.Write(&w.desc, binary.LittleEndian, uint32(len(dims)))
	for _, d := range dims {
		binary.Write(&w.desc, binary.LittleEndian, uint64(d))
	}
	binary.Write(&w.desc, binary.LittleEndian, uint32(typ))
	binary.Write(&w.desc, binary.LittleEndian, off)
	w.nTensor++
}

func (w *writer) bytes() []byte {
	var out bytes.Buffer
	out.WriteString(Magic)
	binary.Write(&out, binary.LittleEndian, uint32(3))
	binary.Write(&out, binary.LittleEndian, w.nTensor)
	binary.Write(&out, binary.LittleEndian, w.nKV)
	out.Write(w.kv.Bytes())
	out.Write(w.desc.Bytes())
	for uint64(out.Len())%w.align != 0 {
		out.WriteByte(0)
	}
	out.Write(w.data.Bytes())
	return out.Bytes()
}

func (w *writer) write(t *testing.T, path string) string {
	t.Helper()
	if err := os.WriteFile(path, w.bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestOpenReadsMetadataAndTensors(t *testing.T) {
	w := newWriter()
	w.putStr(KeyArchitecture, "qwen4exp")
	w.putU32("qwen4exp.block_count", 48)
	w.putU16("small.u16", 65535)
	w.putI32("small.i32", -7)
	w.putF32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6)
	w.putBool("tokenizer.ggml.add_bos_token", true)
	w.putStrArray("tokenizer.ggml.tokens", []string{"<s>", "hello", " world"})
	w.putI32Array("tokenizer.ggml.token_type", []int32{3, 1, 1})
	w.tensor("token_embd.weight", Q8_0, []int64{2560, 64}, 0x11)
	w.tensor("blk.0.attn_norm.weight", F32, []int64{2560}, 0x22)
	w.tensor("blk.0.ffn_gate_exps.weight", Q4_K, []int64{2560, 4, 8}, 0x33)

	path := w.write(t, filepath.Join(t.TempDir(), "m.gguf"))
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if f.Version != 3 {
		t.Errorf("version = %d, want 3", f.Version)
	}
	if got, _ := f.Str(KeyArchitecture); got != "qwen4exp" {
		t.Errorf("architecture = %q", got)
	}
	// Every integer width has to answer to Uint, or a caller would need to
	// know which one the writer picked for a given key.
	for _, k := range []struct {
		key  string
		want uint64
	}{{"qwen4exp.block_count", 48}, {"small.u16", 65535}} {
		if got, ok := f.Uint(k.key); !ok || got != k.want {
			t.Errorf("Uint(%q) = %d, %v; want %d", k.key, got, ok, k.want)
		}
	}
	if got, ok := f.Int("small.i32"); !ok || got != -7 {
		t.Errorf("Int(small.i32) = %d, %v", got, ok)
	}
	if got, ok := f.Float("qwen4exp.attention.layer_norm_rms_epsilon"); !ok || math.Abs(got-1e-6) > 1e-12 {
		t.Errorf("Float(eps) = %g, %v", got, ok)
	}
	if got, ok := f.Bool("tokenizer.ggml.add_bos_token"); !ok || !got {
		t.Errorf("Bool(add_bos) = %v, %v", got, ok)
	}
	if got, ok := f.Strings("tokenizer.ggml.tokens"); !ok || len(got) != 3 || got[2] != " world" {
		t.Errorf("Strings(tokens) = %q, %v", got, ok)
	}
	if got, ok := f.Ints("tokenizer.ggml.token_type"); !ok || len(got) != 3 || got[0] != 3 {
		t.Errorf("Ints(token_type) = %v, %v", got, ok)
	}
	if len(f.Keys()) != 8 {
		t.Errorf("Keys() = %d entries, want 8", len(f.Keys()))
	}

	tn, err := f.Get("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	// Q8_0 is 32 elements in 34 bytes: 2560*64 elements is 5120 blocks.
	if want := int64(2560 * 64 / 32 * 34); int64(len(tn.Data)) != want {
		t.Errorf("token_embd bytes = %d, want %d", len(tn.Data), want)
	}
	if tn.Data[0] != 0x11 || tn.Data[len(tn.Data)-1] != 0x11 {
		t.Errorf("token_embd data does not alias its own span")
	}
	if got := tn.Rows(); got != 64 {
		t.Errorf("Rows = %d, want 64", got)
	}
	if got, want := tn.RowBytes(), int64(2560/32*34); got != want {
		t.Errorf("RowBytes = %d, want %d", got, want)
	}
	if got := tn.Row(63); got[0] != 0x11 || len(got) != int(tn.RowBytes()) {
		t.Errorf("Row(63) is %d bytes", len(got))
	}

	// A 3-D expert bank: rows are the product of every axis but the first,
	// which is what a MoE gather indexes.
	ex, err := f.Get("blk.0.ffn_gate_exps.weight")
	if err != nil {
		t.Fatal(err)
	}
	if got := ex.Rows(); got != 32 {
		t.Errorf("expert Rows = %d, want 32", got)
	}
	if got, want := int64(len(ex.Data)), int64(2560*4*8/256*144); got != want {
		t.Errorf("expert bytes = %d, want %d", got, want)
	}
	if !f.Has("blk.0.attn_norm.weight") || f.Has("nope") {
		t.Error("Has disagrees with the tensor table")
	}
}

func TestTypeBlockGeometry(t *testing.T) {
	// The five formats UD-Q4_K_XL actually uses (LLM.md's inventory), plus
	// the unquantised ones the norms and routers are in. These numbers are
	// the same table reference/gguf_inventory.py carries, and the two
	// agreeing is what makes a Go-side inventory checkable against it.
	for _, c := range []struct {
		typ    Type
		name   string
		elems  int
		nbytes int
	}{
		{F32, "F32", 1, 4}, {F16, "F16", 1, 2}, {BF16, "BF16", 1, 2},
		{Q4_K, "Q4_K", 256, 144}, {Q5_K, "Q5_K", 256, 176},
		{Q5_1, "Q5_1", 32, 24}, {Q8_0, "Q8_0", 32, 34}, {IQ4_NL, "IQ4_NL", 32, 18},
	} {
		if c.typ.String() != c.name {
			t.Errorf("%d.String() = %q, want %q", c.typ, c.typ.String(), c.name)
		}
		if c.typ.BlockElems() != c.elems || c.typ.BlockBytes() != c.nbytes {
			t.Errorf("%s block = %d elems / %d bytes, want %d / %d",
				c.name, c.typ.BlockElems(), c.typ.BlockBytes(), c.elems, c.nbytes)
		}
		n, err := c.typ.SizeOf(int64(c.elems) * 10)
		if err != nil || n != int64(c.nbytes)*10 {
			t.Errorf("%s.SizeOf(10 blocks) = %d, %v", c.name, n, err)
		}
	}
	if _, err := Q4_K.SizeOf(100); err == nil {
		t.Error("SizeOf accepted a partial Q4_K block")
	}
	if _, err := Type(99).SizeOf(32); err == nil {
		t.Error("SizeOf accepted an unknown type")
	}
	if Type(99).String() != "?99" || Type(99).Known() {
		t.Error("an unknown type should report itself as unknown")
	}
	if !Q4_K.Quantised() || F32.Quantised() {
		t.Error("Quantised disagrees with the block geometry")
	}
}

func TestAlignmentIsHonoured(t *testing.T) {
	// general.alignment moves the data section, so a reader that assumed 32
	// would hand back the wrong bytes rather than fail.
	w := newWriter()
	w.align = 4096
	w.putU32(KeyAlignment, 4096)
	w.tensor("a", F32, []int64{8}, 0xAB)
	f, err := Open(w.write(t, filepath.Join(t.TempDir(), "m.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.Alignment != 4096 {
		t.Fatalf("alignment = %d", f.Alignment)
	}
	if f.DataStart%4096 != 0 {
		t.Errorf("data starts at %d, not a 4096 boundary", f.DataStart)
	}
	tn, _ := f.Get("a")
	if tn.Data[0] != 0xAB || len(tn.Data) != 32 {
		t.Errorf("tensor a = %d bytes starting %#x", len(tn.Data), tn.Data[0])
	}
}

func TestRejectsBadFiles(t *testing.T) {
	dir := t.TempDir()
	bad := func(name string, b []byte) error {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		f, err := Open(p)
		if err == nil {
			f.Close()
		}
		return err
	}
	if err := bad("magic.gguf", bytes.Repeat([]byte{0}, 64)); err == nil {
		t.Error("opened a file with no GGUF magic")
	}
	if err := bad("short.gguf", []byte("GGUF")); err == nil {
		t.Error("opened a 4 byte file")
	}

	w := newWriter()
	w.tensor("a", F32, []int64{8}, 1)
	b := w.bytes()
	if err := bad("truncated.gguf", b[:len(b)-16]); err == nil {
		t.Error("opened a file whose last tensor runs past the end")
	}

	w = newWriter()
	w.putU32(KeyAlignment, 48) // not a power of two
	w.tensor("a", F32, []int64{8}, 1)
	if err := bad("align.gguf", w.bytes()); err == nil {
		t.Error("accepted a non-power-of-two alignment")
	}

	// An unknown ggml type has no block geometry, so its size cannot be
	// computed at all; writing one means patching a good descriptor.
	w = newWriter()
	w.tensor("a", F32, []int64{8}, 1)
	if err := bad("unknown-type.gguf", patchLastType(b)); err == nil {
		t.Error("accepted a tensor of an unknown ggml type")
	}
}

// patchLastType rewrites the single tensor descriptor's type field to 99. The
// descriptor is name("a") + n_dims + dims + type + offset, and in the
// one-tensor files above it is the only one, so it can be found by searching
// for the F32 type word followed by a zero offset.
func patchLastType(b []byte) []byte {
	out := append([]byte(nil), b...)
	for i := 24; i+12 <= len(out); i++ {
		if binary.LittleEndian.Uint32(out[i:]) == uint32(F32) && binary.LittleEndian.Uint64(out[i+4:]) == 0 {
			binary.LittleEndian.PutUint32(out[i:], 99)
			return out
		}
	}
	return out
}

func TestOpenSetJoinsShards(t *testing.T) {
	dir := t.TempDir()
	base := "model"
	const n = 3
	for i := 0; i < n; i++ {
		w := newWriter()
		if i == 0 {
			w.putStr(KeyArchitecture, "qwen4exp")
			w.putU32("qwen4exp.expert_count", 512)
		}
		w.putU16(KeySplitNo, uint16(i))
		w.putU16(KeySplitCount, n)
		w.putI32(KeySplitTensors, 6)
		w.tensor(shardTensor(i, 0), Q8_0, []int64{256, 2}, byte(0x10+i))
		w.tensor(shardTensor(i, 1), Q4_K, []int64{256, 2}, byte(0x20+i))
		w.write(t, filepath.Join(dir, shardFile(base, i, n)))
	}

	// Openable from any shard, not only the first: the suffix names the set.
	for _, from := range []string{shardFile(base, 0, n), shardFile(base, 2, n)} {
		s, err := OpenSet(filepath.Join(dir, from))
		if err != nil {
			t.Fatalf("OpenSet(%s): %v", from, err)
		}
		if len(s.Files) != n || s.Len() != 6 {
			t.Errorf("from %s: %d shards, %d tensors", from, len(s.Files), s.Len())
		}
		if got := s.Arch(); got != "qwen4exp" {
			t.Errorf("from %s: arch = %q — metadata should come from shard 1", from, got)
		}
		if got, ok := s.Uint("qwen4exp.expert_count"); !ok || got != 512 {
			t.Errorf("from %s: expert_count = %d, %v", from, got, ok)
		}
		tn, err := s.Get(shardTensor(2, 1))
		if err != nil {
			t.Fatal(err)
		}
		if tn.Shard != 2 || tn.Data[0] != 0x22 {
			t.Errorf("shard-2 tensor landed on shard %d with data %#x", tn.Shard, tn.Data[0])
		}
		s.Close()
	}

	// A directory holding one split set resolves to it.
	s, err := OpenSet(dir)
	if err != nil {
		t.Fatalf("OpenSet(dir): %v", err)
	}
	if len(s.Files) != n {
		t.Errorf("OpenSet(dir) opened %d shards", len(s.Files))
	}
	s.Close()

	// The failure this check exists for: a download that stopped short. Every
	// remaining shard still parses, so nothing but the split keys notices.
	if err := os.Remove(filepath.Join(dir, shardFile(base, 2, n))); err != nil {
		t.Fatal(err)
	}
	if s, err := OpenSet(filepath.Join(dir, shardFile(base, 0, n))); err == nil {
		s.Close()
		t.Error("opened a set with a missing shard")
	}
}

func TestOpenSetRejectsTensorCountMismatch(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 2; i++ {
		w := newWriter()
		w.putU16(KeySplitNo, uint16(i))
		w.putU16(KeySplitCount, 2)
		w.putI32(KeySplitTensors, 5) // the shards hold 2
		w.tensor(shardTensor(i, 0), F32, []int64{8}, byte(i))
		w.write(t, filepath.Join(dir, shardFile("m", i, 2)))
	}
	if s, err := OpenSet(filepath.Join(dir, shardFile("m", 0, 2))); err == nil {
		s.Close()
		t.Error("accepted a set whose split.tensors.count disagrees with the tables")
	}
}

func shardFile(base string, i, n int) string {
	return base + "-" + pad5(i+1) + "-of-" + pad5(n) + ".gguf"
}

func shardTensor(shard, i int) string {
	return "blk." + pad5(shard) + ".w" + pad5(i)
}

func pad5(n int) string {
	s := ""
	for _, d := range []int{10000, 1000, 100, 10, 1} {
		s += string(rune('0' + (n/d)%10))
	}
	return s
}
