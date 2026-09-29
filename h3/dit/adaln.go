package dit

import (
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/zimage/qwen"
)

// TablesCached is Tables kept in cacheDir, keyed by the checkpoint and the
// timesteps (VIDEO.md M11b). A request's timesteps are a function of its
// step count and whether it has keyframes, so a server sees a handful of
// sets, and each is ~0.8 GB of fp32 read back in a second where computing
// it reads 17 GB of bf16 (~20 s). cacheDir "" is Tables.
func TablesCached(dir string, tvals []float32, cacheDir string) ([]*Table, error) {
	if cacheDir == "" {
		return Tables(dir, tvals)
	}
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	var key strings.Builder
	key.WriteString("h3 adaln tables")
	for _, t := range tvals {
		fmt.Fprintf(&key, " %08x", math.Float32bits(t))
	}
	path := filepath.Join(cacheDir, fmt.Sprintf("adaln-%s.bin", qwen.CacheKey(dir, key.String())))
	per := len(tvals) * Modalities * 6 * cfg.Hidden
	if tabs, ok := readTables(path, cfg.Layers, per, len(tvals)*Modalities, cfg.Hidden); ok {
		return tabs, nil
	}
	tabs, err := Tables(dir, tvals)
	if err != nil {
		return nil, err
	}
	if err := writeTables(path, tabs); err != nil {
		log.Printf("dit: not caching the AdaLN tables: %v", err)
	}
	return tabs, nil
}

func tableBytes(t *Table) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(t.Data))), 4*len(t.Data))
}

// readTables reads layers tables of per values each; any short or missing
// file is a miss.
func readTables(path string, layers, per, rows, hidden int) ([]*Table, bool) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false
	}
	defer f.Close()
	if st, err := f.Stat(); err != nil || st.Size() != int64(layers*per*4) {
		return nil, false
	}
	tabs := make([]*Table, layers)
	for i := range tabs {
		tabs[i] = &Table{Hidden: hidden, Rows: rows, Data: make([]float32, per)}
		if _, err := io.ReadFull(f, tableBytes(tabs[i])); err != nil {
			return nil, false
		}
	}
	return tabs, true
}

// writeTables stores tabs through a temporary file, so a reader never sees
// part of one.
func writeTables(path string, tabs []*Table) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".tmp-adaln-*")
	if err != nil {
		return err
	}
	for _, t := range tabs {
		if _, err = f.Write(tableBytes(t)); err != nil {
			break
		}
	}
	if err == nil {
		err = f.Chmod(0o644)
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(f.Name(), path)
	}
	if err != nil {
		os.Remove(f.Name())
	}
	return err
}

// Tables computes every block's AdaLN table for a request's distinct
// timesteps tvals, in fp32 on the host from the checkpoint's bf16
// projections: 13 B of the transformer's 33 B parameters, applied once per
// request and never staged on the device (VIDEO.md decision 3). Each block's
// projection is read, applied and dropped in turn, so the host holds one
// block's (1 GB as fp32) at a time.
func Tables(dir string, tvals []float32) ([]*Table, error) {
	host, err := LoadHost(dir)
	if err != nil {
		return nil, err
	}
	temb, err := host.TimeEmbed(tvals)
	if err != nil {
		return nil, err
	}
	act := siluMat(temb)
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	c := host.Cfg
	tabs := make([]*Table, c.Layers)
	for i := range tabs {
		runtime.GC() // the last projection's 1 GB of fp32, beside the encoder's staging
		l := &loader{set: set}
		lin := l.linear(fmt.Sprintf("transformer_blocks.%d.adaln_proj.linear", i), 6*Modalities*c.Hidden, c.TimeDim, true)
		if l.err != nil {
			return nil, l.err
		}
		out, err := lin.Apply(act)
		if err != nil {
			return nil, err
		}
		tabs[i] = &Table{Hidden: c.Hidden, Rows: temb.Rows * Modalities, Data: out.Data}
	}
	return tabs, nil
}
