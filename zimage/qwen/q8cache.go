package qwen

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"strix-halo-vulkan/vk"
)

// q8CacheVersion is part of every key. Bump it when PackQ8, or a staging's
// placement of its matrices in the banks, changes what it writes for the
// same checkpoint: an old file is then never looked for.
const q8CacheVersion = 1

// CacheKey is a short hash of parts and of the checkpoint in src: every
// shard's name, size and modification time. It does not hash the shards'
// bytes (tens of GB, the cost the cache exists to avoid), so a checkpoint
// rewritten in place with the same sizes and times is the case it cannot
// see. A checkpoint re-downloaded or copied is a miss, and is re-packed.
func CacheKey(src string, parts ...string) string {
	h := sha256.New()
	fmt.Fprintf(h, "v%d\n", q8CacheVersion)
	for _, p := range parts {
		fmt.Fprintf(h, "%s\n", p)
	}
	shards, _ := filepath.Glob(filepath.Join(src, "*.safetensors"))
	for _, p := range shards {
		if st, err := os.Stat(p); err == nil {
			fmt.Fprintf(h, "%s %d %d\n", filepath.Base(p), st.Size(), st.ModTime().UnixNano())
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Q8Cache keeps one staging's int8 banks on disk, byte for byte as the
// device holds them, one file a bank (VIDEO.md M11b). Packing is the cost it
// removes: a bank is a pure function of the checkpoint and its layout, and
// re-reading bf16, widening it to fp32 and quantising it was ~70 s of every
// MiniMax-H3 request, where the files read back at disk speed straight into
// the mapped banks.
//
// A nil *Q8Cache is a cache that is never there: Load misses, and Put,
// Commit and Abort do nothing.
type Q8Cache struct {
	paths []string
	sizes []int
	tmp   []*os.File
	err   error
}

// OpenQ8Cache names the files of one staging's banks: in dir, for the
// checkpoint in src, under layout (everything else the bytes depend on:
// which matrices, their shapes and where they sit), one of each size in
// sizes. dir "" is no cache.
func OpenQ8Cache(dir, src, layout string, sizes []int) *Q8Cache {
	if dir == "" || len(sizes) == 0 {
		return nil
	}
	key := CacheKey(src, layout)
	c := &Q8Cache{sizes: sizes}
	for i := range sizes {
		c.paths = append(c.paths, filepath.Join(dir, fmt.Sprintf("q8-%s-%d.bin", key, i)))
	}
	return c
}

// q8ReadChunk is one reader's share of a file at a time. Several in flight
// are what reach the disk's rate: one reader is bound by the copy (and
// dm-crypt's decryption) on one core.
const (
	q8ReadChunk   = 64 << 20
	q8ReadWorkers = 8
)

// Load fills bufs from the files and reports whether it did. Every file has
// to be there at its size; anything else is a miss, which the staging that
// follows repairs by writing them anew.
func (c *Q8Cache) Load(bufs []*vk.Buffer) bool {
	if c == nil || len(bufs) != len(c.paths) {
		return false
	}
	files := make([]*os.File, len(c.paths))
	defer func() {
		for _, f := range files {
			if f != nil {
				f.Close()
			}
		}
	}()
	type chunk struct{ file, off, n int }
	var work []chunk
	for i, p := range c.paths {
		st, err := os.Stat(p)
		if err != nil || st.Size() != int64(c.sizes[i]) || bufs[i].Size() != c.sizes[i] || !bufs[i].Mapped() {
			return false
		}
		if files[i], err = os.Open(p); err != nil {
			return false
		}
		for off := 0; off < c.sizes[i]; off += q8ReadChunk {
			work = append(work, chunk{i, off, min(q8ReadChunk, c.sizes[i]-off)})
		}
	}
	jobs := make(chan chunk)
	var mu sync.Mutex
	failed := false
	var wg sync.WaitGroup
	for range q8ReadWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for w := range jobs {
				dst := unsafe.Slice((*byte)(bufs[w.file].MappedPointer()), bufs[w.file].Size())
				if _, err := files[w.file].ReadAt(dst[w.off:w.off+w.n], int64(w.off)); err != nil {
					mu.Lock()
					failed = true
					mu.Unlock()
				}
			}
		}()
	}
	for _, w := range work {
		jobs <- w
	}
	close(jobs)
	wg.Wait()
	return !failed
}

// Put records b as written at byte off of bank i: on a miss, the staging
// passes every matrix it packs through here as it writes it to the device.
// The first failure (a full disk) drops the cache for this staging and
// nothing else: it costs the next request time, never this one.
func (c *Q8Cache) Put(bank, off int, b []byte) {
	if c == nil || c.err != nil {
		return
	}
	if c.tmp == nil {
		c.tmp = make([]*os.File, len(c.paths))
		if c.err = os.MkdirAll(filepath.Dir(c.paths[0]), 0o755); c.err != nil {
			c.drop()
			return
		}
		for i, p := range c.paths {
			if c.tmp[i], c.err = os.CreateTemp(filepath.Dir(p), ".tmp-q8-*"); c.err != nil {
				c.drop()
				return
			}
		}
	}
	if _, c.err = c.tmp[bank].WriteAt(b, int64(off)); c.err != nil {
		c.drop()
	}
}

// Commit moves the written files into place, sized to their banks (the
// padding between matrices is never written). A staging calls it once
// every matrix has been Put.
func (c *Q8Cache) Commit() {
	if c == nil || c.tmp == nil || c.err != nil {
		return
	}
	for i, f := range c.tmp {
		if c.err = f.Truncate(int64(c.sizes[i])); c.err == nil {
			c.err = f.Chmod(0o644) // CreateTemp's 0600, as the checkpoint's files are not
		}
		if c.err != nil {
			c.drop()
			return
		}
	}
	for i, f := range c.tmp {
		if c.err = f.Close(); c.err == nil {
			c.err = os.Rename(f.Name(), c.paths[i])
		}
		if c.err != nil {
			c.drop()
			return
		}
		c.tmp[i] = nil
	}
	total := 0
	for _, n := range c.sizes {
		total += n
	}
	log.Printf("qwen: int8 banks cached, %.1f GB in %s", float64(total)/1e9, filepath.Dir(c.paths[0]))
	c.tmp = nil
}

// Abort removes whatever a staging that failed part-way had written.
func (c *Q8Cache) Abort() {
	if c == nil || c.tmp == nil {
		return
	}
	c.drop()
}

func (c *Q8Cache) drop() {
	if c.err != nil {
		log.Printf("qwen: not caching the int8 banks: %v", c.err)
	}
	for _, f := range c.tmp {
		if f != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}
	c.tmp = nil
	if c.err == nil {
		c.err = os.ErrClosed
	}
}
