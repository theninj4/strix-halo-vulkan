package qwen

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"strix-halo-vulkan/vk"
)

// TestGPUQ8Cache round-trips two banks through the cache and checks the
// three ways it must miss: nothing written yet, a bank of another size, and
// a checkpoint shard that changed.
func TestGPUQ8Cache(t *testing.T) {
	dev, done := newTestDevice(t)
	t.Cleanup(done)
	src, dir := t.TempDir(), filepath.Join(t.TempDir(), "bank-cache")
	shard := filepath.Join(src, "model.safetensors")
	if err := os.WriteFile(shard, []byte("stand-in"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Two banks, the second spanning several read chunks and a tail.
	sizes := []int{4096, 3*q8ReadChunk + 768}
	newBanks := func() []*vk.Buffer {
		var bufs []*vk.Buffer
		for _, n := range sizes {
			b, err := dev.NewBuffer(n)
			if err != nil {
				t.Fatal(err)
			}
			b.Zero()
			t.Cleanup(b.Destroy)
			bufs = append(bufs, b)
		}
		return bufs
	}
	view := func(b *vk.Buffer) []byte { return unsafe.Slice((*byte)(b.MappedPointer()), b.Size()) }

	c := OpenQ8Cache(dir, src, "layout", sizes)
	src1 := newBanks()
	if c.Load(src1) {
		t.Fatal("an empty cache hit")
	}
	// Written as a staging writes: matrices at offsets, the padding never.
	want := make([][]byte, len(sizes))
	for i, n := range sizes {
		want[i] = make([]byte, n)
		for j := range want[i][:n-256] {
			want[i][j] = byte(j*7 + i + 1)
		}
		half := (n - 256) / 2
		c.Put(i, half, want[i][half:n-256])
		c.Put(i, 0, want[i][:half])
	}
	c.Commit()

	got := newBanks()
	if !OpenQ8Cache(dir, src, "layout", sizes).Load(got) {
		t.Fatal("a committed cache missed")
	}
	for i := range sizes {
		if !bytes.Equal(view(got[i]), want[i]) {
			t.Errorf("bank %d came back different", i)
		}
	}
	if OpenQ8Cache(dir, src, "another layout", sizes).Load(newBanks()) {
		t.Error("another layout hit")
	}
	other := []int{sizes[0], sizes[1] + 256}
	if OpenQ8Cache(dir, src, "layout", other).Load(newBanks()) {
		t.Error("a bank of another size hit")
	}
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(shard, later, later); err != nil {
		t.Fatal(err)
	}
	if OpenQ8Cache(dir, src, "layout", sizes).Load(newBanks()) {
		t.Error("a changed checkpoint hit")
	}
	if OpenQ8Cache("", src, "layout", sizes).Load(newBanks()) {
		t.Error("no cache hit")
	}
	if left, _ := filepath.Glob(filepath.Join(dir, ".tmp-*")); len(left) != 0 {
		t.Errorf("temporary files left: %v", left)
	}
}
