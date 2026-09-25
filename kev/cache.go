package kev

// The prefix cache (CLASSIFICATION.md K7.2): a state's KV and recurrent state,
// kept across requests, keyed by its token ids.
//
// A System One caller asks several questions of one text, often over several
// requests -- a new question about a ticket already classified -- and the
// state is most of the tokens. Kev's server keeps the last four states' caches
// and a repeated state pays for its questions only ("same text again" in its
// serving table). This is that, with the same default of four.
//
// What a state leaves behind is small and fixed in the recurrent layers and
// grows with its length only in the eight attention layers:
//
//	24 GDN layers   S [32][128][128] + conv tail [3][8192]     52.7 MB, any length
//	 8 attention    K and V, fp16 [Ls][4][256] each            32 KB a token
//
// The working copy lives in the activation arena, where the kernels read it;
// a slot is a copy of it in the cache's own buffer. Saving and restoring are
// device copies (kev_copy.comp), never a host round trip: the host reads these
// mapped arenas at ~0.2 GB/s, and a slot is ~60 MB for a short state.
//
// Slots are fixed-size, sized for CacheTokens cells of KV (32 KB a cell since K7.3's fp16 cache). A longer state is
// not cached across requests; it still runs as one state pass and then its
// branches against the working copy, which is what the cache is for within
// one request.

import (
	"fmt"
	"slices"
	"time"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

type prefixCache struct {
	buf   *vk.Buffer
	pipe  *vk.ComputePipeline
	mod   *vk.ShaderModule
	cells int // KV cells a slot holds: the longest state it caches
	slots []cacheSlot
	tick  uint64

	gdnFloats  int // the GDN block's floats
	slotFloats int

	Hits, Misses, Saves int
}

type cacheSlot struct {
	key  []int32
	used uint64
}

// newPrefixCache allocates n slots of `cells` KV cells each. n = 0 is no cache.
func newPrefixCache(g *GPU, n, cells int) (*prefixCache, error) {
	c := &prefixCache{cells: cells, gdnFloats: g.gdnFloats}
	if n <= 0 || cells <= 0 {
		return c, nil
	}
	attn := 0
	for _, lw := range g.w {
		if lw.attn {
			attn++
		}
	}
	// K and V are fp16 since K7.3: a cell of one layer is kvRow/2 words each.
	c.slotFloats = g.gdnFloats + attn*cells*kvRow
	if c.slotFloats*n*4 > maxBankBytes {
		return nil, fmt.Errorf("kev: %d cache slots of %d tokens are %d MB, over one buffer's 4 GiB", n, cells, (c.slotFloats*n*4)>>20)
	}
	var err error
	if c.buf, err = g.dev.NewBuffer(c.slotFloats * n * 4); err != nil {
		return nil, fmt.Errorf("kev: prefix cache (%d MB): %w", (c.slotFloats*n*4)>>20, err)
	}
	if c.mod, err = g.dev.NewShaderModule(shaders.KevCopy); err != nil {
		return nil, err
	}
	c.pipe, err = g.dev.NewPipeline(c.mod, vk.PipelineSpec{
		Buffers:          []*vk.Buffer{g.wbuf, g.abuf, g.hbuf, g.banks[0], c.buf},
		PushConstantSize: pushBytes,
	})
	if err != nil {
		return nil, err
	}
	c.slots = make([]cacheSlot, n)
	return c, nil
}

func (c *prefixCache) destroy() {
	if c == nil {
		return
	}
	if c.pipe != nil {
		c.pipe.Destroy()
	}
	if c.mod != nil {
		c.mod.Destroy()
	}
	if c.buf != nil {
		c.buf.Destroy()
	}
}

// kvRow is one cell of one attention layer's K (or V): 4 heads of 256.
const kvRow = 4 * 256

// fits is whether a state of ls tokens can be cached.
func (c *prefixCache) fits(ls int) bool { return c != nil && len(c.slots) > 0 && ls <= c.cells }

// lookup returns the slot holding exactly this state, or -1, and counts it.
func (c *prefixCache) lookup(key []int32) int {
	i := c.peek(key)
	c.count(i >= 0)
	return i
}

// peek is lookup without the count (a batch counts once it knows it runs as
// one); it does mark the slot used.
func (c *prefixCache) peek(key []int32) int {
	if c == nil || len(c.slots) == 0 {
		return -1
	}
	for i := range c.slots {
		if c.slots[i].key != nil && slices.Equal(c.slots[i].key, key) {
			c.tick++
			c.slots[i].used = c.tick
			return i
		}
	}
	return -1
}

func (c *prefixCache) count(hit bool) {
	if c == nil || len(c.slots) == 0 {
		return
	}
	if hit {
		c.Hits++
	} else {
		c.Misses++
	}
}

// copies is the dispatch list moving a working state -- GDN slot `work`, and
// attention cells [sLo, sLo+ls) -- to or from cache slot `slot`: the GDN
// slot whole, then each attention layer's K and V cells.
func (c *prefixCache) copies(g *GPU, slot, work, sLo, ls int, restore bool) []vk.MultiDispatch {
	var d []vk.MultiDispatch
	at := slot * c.slotFloats
	// move copies n words; half says the arena side is the fp16 one, whose
	// offsets are in halves and so are halved into words.
	move := func(arena uint32, cache, n int, half bool) {
		pc := push{Dim: uint32(n)}
		dir := uint32(0)
		if half {
			arena /= 2
			dir = 2
		}
		if restore {
			pc.InOff, pc.OutOff, pc.Aux0 = uint32(cache), arena, dir+1
		} else {
			pc.InOff, pc.OutOff, pc.Aux0 = arena, uint32(cache), dir
		}
		d = append(d, vk.MultiDispatch{Pipeline: c.pipe, GroupsX: min(groups(n, 256), 1024), GroupsY: 1, PushConstants: pc.bytes()})
	}
	move(g.gdnBlock+uint32(work*g.gdnFloats), at, c.gdnFloats, false)
	at += c.gdnFloats
	for _, lw := range g.w {
		if !lw.attn {
			continue
		}
		move(lw.kc+uint32(sLo*kvRow), at, ls*kvRow/2, true)
		move(lw.vc+uint32(sLo*kvRow), at+c.cells*kvRow/2, ls*kvRow/2, true)
		at += c.cells * kvRow
	}
	return d
}

// save copies a working state (GDN slot `work`, cells from sLo) into the
// least recently used cache slot.
func (c *prefixCache) save(g *GPU, key []int32, work, sLo int) (time.Duration, error) {
	victim := 0
	for i := range c.slots {
		if c.slots[i].key == nil {
			victim = i
			break
		}
		if c.slots[i].used < c.slots[victim].used {
			victim = i
		}
	}
	dur, err := vk.DispatchMultiTimed(c.copies(g, victim, work, sLo, len(key), false), 1, 1, true)
	if err != nil {
		return 0, fmt.Errorf("kev: saving a state: %w", err)
	}
	c.tick++
	c.slots[victim] = cacheSlot{key: slices.Clone(key), used: c.tick}
	c.Saves++
	return dur, nil
}

// restore copies cache slot `slot` into GDN slot `work` and attention cells
// [sLo, sLo+ls).
func (c *prefixCache) restore(g *GPU, slot, work, sLo, ls int) (time.Duration, error) {
	dur, err := vk.DispatchMultiTimed(c.copies(g, slot, work, sLo, ls, true), 1, 1, true)
	if err != nil {
		return 0, fmt.Errorf("kev: restoring a state: %w", err)
	}
	return dur, nil
}

// clear forgets every slot.
func (c *prefixCache) clear() {
	if c == nil {
		return
	}
	for i := range c.slots {
		c.slots[i] = cacheSlot{}
	}
}

// CacheStats reports the prefix cache's counters.
func (g *GPU) CacheStats() (hits, misses, saves, slots, tokens int) {
	c := g.cache
	if c == nil {
		return
	}
	return c.Hits, c.Misses, c.Saves, len(c.slots), c.cells
}

// ClearCache forgets every cached state.
func (g *GPU) ClearCache() { g.cache.clear() }
