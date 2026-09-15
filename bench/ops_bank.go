package bench

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
	"unsafe"

	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// The weight-bank gather probe — LLM.md **L0a**, and the one measurement the
// qwen3.8-flash-next vertical is not allowed to start without.
//
// Everything LLM.md concludes about quantisation rests on one number: that a
// 4-bit decode step reads its weights at ~242 GB/s, so tok/s is linear in
// bits/weight and nothing else. That number comes from §1.1 and §1.7, and
// both measured it over a *small* range — `gemv_cold`'s largest DRAM-resident
// case reads 64 MB of 4-bit weights, `bandwidth` tops out at a 512 MB
// footprint, and `moe`'s expert banks reach 4 GB. The model's resident bank
// is **64-82 GB**, read as ~500 scattered expert slabs per token. That is 20x
// the largest range anything here has probed, and §5.2 ("heap topology,
// carveout size and page size ... informs how to lay out a big model") has
// never been run.
//
// So: hold the bytes read per step fixed, hold the access shape fixed, and
// vary only **the size of the address range they are drawn from**. If GB/s is
// flat from 1 GiB to 64 GiB, the bus survives a big model and LLM.md's
// arithmetic stands. If it falls, the shape of the fall is the actionable
// result — it prices a smaller resident bank against a wider format.
//
// Three orders separate two things a naive "read a big bank" test conflates:
//
//	seq     the first K slabs of the prefix, contiguous. The control: the
//	        range is nominally large but the working set is not, so this
//	        should read the same at every prefix and any drift in it is
//	        measurement noise rather than a finding.
//	spread  K slabs evenly spaced across the whole prefix, ascending. The
//	        range is real but the order is not random.
//	rand    K slabs drawn at random from the whole prefix. What MoE decode
//	        actually does: ten of 512 experts per layer, chosen by a router.
//
// At the smallest prefix all three select the same set, which is a free
// internal consistency check — they must agree there or the harness is wrong.
//
// What this deliberately does not do is run the real GEMV. The kernel is
// already at 99-103% of the bus (§1.7); if the memory system delivers the
// bytes, the kernel will consume them, and standing a tuned GEMV in the way
// of a memory-system question only adds a variable. See bank_gather.comp.

// bankBufferBytes is one allocation's size. maxMemoryAllocationSize on this
// device is 0xfffffffc — 4 GiB minus 4 — so a multi-gigabyte bank is
// necessarily many buffers, and 3.5 GiB leaves room for the driver's
// rounding of VkMemoryRequirements.size.
const bankBufferBytes = 7 << 29 // 3.5 GiB

// bankPageBytes is the granularity the bank is dirtied at before it is
// measured. An allocation whose pages have never been written may not have
// distinct physical pages behind it, which would turn the whole experiment
// into a measurement of one cached page; touching every page is what makes
// the working set real. It is also, and not incidentally, what makes the
// page-table footprint real, which is the thing under test.
const bankPageBytes = 4096

// bankWG must match bank_gather.comp's local_size_x.
const bankWG = 256

// BankSlabBytesList is the swept slab size: the contiguous run one gather
// step reads. The first two are qwen3.8-flash-next's real 4-bit expert
// slabs — `down` is 2560x640 nibbles (819200 B) and `gate_up` is 1280x2560
// (1638400 B) — and 4 MiB is a control an order of magnitude up, to show
// whether any effect found is about the slab or about the range.
var BankSlabBytesList = []int{819200, 1638400, 4 << 20}

// BankPrefixesGiB is the swept working-set size, in GiB. It runs from well
// under anything interesting up to the whole device-local heap's worth: the
// UD-Q4_K_XL resident core is 82.5 GB and a bank of our own would be ~67 GB,
// so 64 GiB (68.7 GB) is the size the answer actually has to hold at.
// 80 GiB (85.9 GB) is only reached by an explicit -bankgib 80: it covers
// UD-Q4_K_XL's 82.52 GB resident core with margin, and unlike L0b's capacity
// probe every page of it is written, so it measures residency rather than
// reservation.
var BankPrefixesGiB = []int{1, 2, 4, 8, 16, 32, 48, 64, 80}

// bankOrders are the three slab selections, in the order they are reported.
var bankOrders = []string{"seq", "spread", "rand"}

// bankBuf is one allocation of the bank plus the per-buffer slab table and
// pipeline that address it. The pipelines are built once and reused across
// every cell: only the table's *contents* and the push constants change, so
// a sweep costs no pipeline churn.
type bankBuf struct {
	src   *vk.Buffer
	slabs *vk.Buffer
	pipe  *vk.ComputePipeline
	off   int // this buffer's first byte, as an offset into the whole bank
	size  int
}

// RunBank measures achieved read bandwidth at a fixed number of bytes and a
// fixed access shape, sweeping only the size of the range those bytes are
// drawn from.
func RunBank(dev *vk.Device, phys *vk.PhysicalDevice, p Params, out io.Writer) ([]Result, error) {
	mod, err := dev.NewShaderModule(shaders.BankGather)
	if err != nil {
		return nil, err
	}
	defer mod.Destroy()

	readBytes := p.BankReadMiB << 20
	if readBytes <= 0 {
		return nil, fmt.Errorf("bank: -bankreadmib must be positive")
	}

	// The dst buffer is shared by every dispatch; each buffer's dispatch is
	// given a disjoint dstBase so nothing collides. One dword per workgroup,
	// and the widest cell is readBytes/min(slab) slabs times parts.
	maxSlabs := readBytes/BankSlabBytesList[0] + 1
	for _, s := range BankSlabBytesList {
		if n := readBytes/s + 1; n > maxSlabs {
			maxSlabs = n
		}
	}
	dst, err := dev.NewBuffer(maxSlabs * bankMaxParts * 4)
	if err != nil {
		return nil, fmt.Errorf("bank: dst buffer: %w", err)
	}
	defer dst.Destroy()

	bufs, err := allocBank(dev, mod, dst, p.BankGiB<<30, maxSlabs, bankDefaultType, out)
	for _, b := range bufs {
		defer b.release()
	}
	if err != nil {
		return nil, err
	}
	if len(bufs) == 0 {
		return nil, fmt.Errorf("bank: could not allocate any of the requested %d GiB", p.BankGiB)
	}
	allocated := bufs[len(bufs)-1].off + bufs[len(bufs)-1].size
	fmt.Fprintf(out, "bank: %d buffers, %.1f GiB allocated of %d requested\n",
		len(bufs), float64(allocated)/(1<<30), p.BankGiB)

	if err := verifyBankGather(dev, bufs[0], dst); err != nil {
		return nil, fmt.Errorf("bank: %w", err)
	}

	var results []Result
	for _, slabBytes := range BankSlabBytesList {
		parts, ok := bankParts(slabBytes)
		if !ok {
			return nil, fmt.Errorf("bank: slab %d B has no legal part count", slabBytes)
		}
		chunksPerPart := slabBytes / 16 / parts
		slabCount := readBytes / slabBytes
		if slabCount == 0 {
			continue
		}
		for _, gib := range BankPrefixesGiB {
			prefix := gib << 30
			if prefix > allocated {
				continue
			}
			slots := bankSlots(bufs, prefix, slabBytes)
			if slots.total < slabCount {
				continue
			}
			for _, order := range bankOrders {
				pick := selectSlabs(slots, slabCount, order)
				dispatches := loadSlabTables(bufs, pick, slabBytes, chunksPerPart, parts)
				ns, clocks, err := TimeDispatchMulti(dispatches, 1, p.Warmup, p.Iters, false)
				if err != nil {
					return nil, fmt.Errorf("bank slab=%d prefix=%dGiB %s: %w", slabBytes, gib, order, err)
				}
				results = append(results, Result{
					Op: "bank", Variant: order, Size: gib,
					Detail: fmt.Sprintf("slabKB=%d;readMiB=%d;bufs=%d;parts=%d",
						slabBytes/1024, readBytes>>20, len(dispatches), parts),
					NsPerIter: ns,
					GBPS:      float64(readBytes) / (ns / 1e9) / 1e9,
					Clocks:    clocks,
				})
			}
		}
	}

	// L0b (§5.1) runs after the range sweep so that it inherits the same
	// verified kernel and the same dst buffer, and so that the range result
	// is already in hand when the memory types are compared against it.
	typeResults, err := runBankMemTypes(dev, mod, dst, p, maxSlabs, out)
	if err != nil {
		return results, err
	}
	results = append(results, typeResults...)

	if p.BankHeadroomGiB > 0 {
		// Off by default: it allocates until the driver refuses, which is
		// not something to do to a workstation without being asked.
		if err := RunBankHeadroom(dev, p.BankHeadroomType, p.BankHeadroomGiB, out); err != nil {
			return results, err
		}
	}
	return results, nil
}

// bankMemTypeGiB is how much of each memory type the L0b arm allocates. It has
// to clear two thresholds to mean anything: the 32 MiB MALL, or the probe
// would measure cache, and the ~8 GB at which `cmd/bus` found host reads of
// the preferred type change character — the point where this part's real
// 8 GiB visible-VRAM carveout runs out and something else starts backing the
// allocation. 12 GiB clears both.
const bankMemTypeGiB = 12

// bankMemTypePrefixesGiB are measured within each type: one comfortably
// inside the carveout and one well past it. If the carveout boundary matters
// to a *GPU* read the way it matters to a host read, it shows up as these two
// disagreeing.
var bankMemTypePrefixesGiB = []int{1, 12}

// runBankMemTypes is LLM.md **L0b** and IDEAS **§5.1**, never run until now.
//
// Every number in the range sweep above came from whatever vk.NewBuffer
// prefers, which is memory type 3 — DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT,
// heap 1, and heap 1 stops at 83.79 GiB. UD-Q4_K_XL's resident core is
// 82.52 GB, which is 96% of that, so whether the *other* heap reads as fast
// decides whether a model larger than heap 1 can be served at all — and
// therefore whether anything wider than ~4.25 bits/weight is on the table.
//
// §5.1 asked the more general question ("which memory type is fastest for
// GPU-read-only weights?") and predicted the answer would be worth a few
// percent on everything. This arm answers both at once: it walks every type
// that can back a storage buffer, allocates the same bank from each and runs
// the same gather over it.
func runBankMemTypes(dev *vk.Device, mod *vk.ShaderModule, dst *vk.Buffer, p Params, maxSlabs int, out io.Writer) ([]Result, error) {
	types, err := dev.MemoryTypes()
	if err != nil {
		return nil, fmt.Errorf("bank: memory types: %w", err)
	}
	fmt.Fprintf(out, "\nbank/L0b: %d memory types\n", len(types))
	for _, t := range types {
		fmt.Fprintf(out, "  type %2d  heap %d  %5.1f GiB  buffer=%v  %s\n",
			t.Index, t.HeapIndex, float64(t.HeapSize)/(1<<30), t.BufferCompatible, t)
	}

	const slabBytes = 1638400 // the model's gate_up expert slab at 4 bits
	parts, ok := bankParts(slabBytes)
	if !ok {
		return nil, fmt.Errorf("bank: slab %d B has no legal part count", slabBytes)
	}
	chunksPerPart := slabBytes / 16 / parts
	readBytes := p.BankReadMiB << 20
	slabCount := readBytes / slabBytes

	var results []Result
	for _, t := range types {
		if !t.BufferCompatible {
			continue
		}
		rs, err := bankOneMemType(dev, mod, dst, t, maxSlabs, slabBytes, parts, chunksPerPart, slabCount, readBytes, p, out)
		if err != nil {
			return results, err
		}
		results = append(results, rs...)
	}
	return results, nil
}

func bankOneMemType(dev *vk.Device, mod *vk.ShaderModule, dst *vk.Buffer, t vk.MemoryType,
	maxSlabs, slabBytes, parts, chunksPerPart, slabCount, readBytes int, p Params, out io.Writer) ([]Result, error) {

	bufs, err := allocBank(dev, mod, dst, bankMemTypeGiB<<30, maxSlabs, t.Index, out)
	defer func() {
		for _, b := range bufs {
			b.release()
		}
	}()
	if err != nil || len(bufs) == 0 {
		fmt.Fprintf(out, "bank/L0b: type %d gave nothing\n", t.Index)
		return nil, nil //nolint:nilerr // an unusable type is a result, not a failure
	}
	allocated := bufs[len(bufs)-1].off + bufs[len(bufs)-1].size
	mapped := bufs[0].src.Mapped()

	var results []Result
	for _, gib := range bankMemTypePrefixesGiB {
		prefix := gib << 30
		if prefix > allocated {
			continue
		}
		slots := bankSlots(bufs, prefix, slabBytes)
		if slots.total < slabCount {
			continue
		}
		dispatches := loadSlabTables(bufs, selectSlabs(slots, slabCount, "rand"), slabBytes, chunksPerPart, parts)
		ns, clocks, err := TimeDispatchMulti(dispatches, 1, p.Warmup, p.Iters, false)
		if err != nil {
			return results, fmt.Errorf("bank memtype=%d prefix=%dGiB: %w", t.Index, gib, err)
		}
		results = append(results, Result{
			Op: "bank", Variant: "memtype", Size: gib,
			Detail: fmt.Sprintf("type=%d;heap=%d;gotGiB=%.0f;mapped=%v;flags=%s",
				t.Index, t.HeapIndex, float64(allocated)/(1<<30), mapped, t),
			NsPerIter: ns,
			GBPS:      float64(readBytes) / (ns / 1e9) / 1e9,
			Clocks:    clocks,
		})
	}
	return results, nil
}

// bankHeadroomFloorGiB is how much host memory the headroom probe refuses to
// go below. It is allocating without writing, so the pages should stay
// uncommitted and nothing should be at risk — but "should" is doing work in
// that sentence on a driver that hands out GTT, and this runs on the
// developer's own workstation. The probe stops rather than finds out.
const bankHeadroomFloorGiB = 12

// RunBankHeadroom answers the capacity half of L0b, which is a different
// question from the bandwidth half above: **how much will the device actually
// hand out?**
//
// It matters because the numbers are close and the reported ones turn out to
// be fiction. RADV reports heap 1 (device-local) at 83.79 GiB and heap 0 at
// 41.89 GiB on a machine with 117.7 GiB of GTT. UD-Q4_K_XL's resident core is
// 82.52 GB = 76.9 GiB, which is 92% of heap 1, and a 262 k KV cache is another
// 6.4 GB on top — so whether "83.79 GiB" is a wall decides whether the stock
// quant can be served at all, and whether anything wider than ~4.25 bits is
// worth considering.
//
// **One probe per process, deliberately.** Freeing a Vulkan allocation returns
// its pages to the driver's TTM pool rather than to the kernel's free list:
// after this probe releases 63 GiB, `MemAvailable` stays where it was and
// `mem_info_gtt_used` drops to ~90 MB. So a second probe in the same process
// sees a machine that looks full and is not, and reports zero. Measuring two
// memory types means two runs.
//
// Allocation only — no page is written. That measures what the driver will
// reserve, which is an upper bound on what the machine will hold; the largest
// bank this suite has actually *touched* is the 64 GiB of the range sweep.
func RunBankHeadroom(dev *vk.Device, memType uint32, capGiB int, out io.Writer) error {
	types, err := dev.MemoryTypes()
	if err != nil {
		return err
	}
	if memType == bankDefaultType {
		for _, t := range types {
			if t.BufferCompatible && t.Has(vk.MemoryDeviceLocal|vk.MemoryHostVisible|vk.MemoryHostCoherent) {
				memType = t.Index
				break
			}
		}
	}
	if int(memType) >= len(types) || !types[memType].BufferCompatible {
		return fmt.Errorf("bank: memory type %d cannot back a storage buffer", memType)
	}
	t := types[memType]

	var bufs []*vk.Buffer
	defer func() {
		for _, b := range bufs {
			b.Destroy()
		}
	}()
	got, why := fillWith(dev, memType, capGiB<<30, &bufs)

	fmt.Fprintf(out, "\nL0b capacity — how much will the device actually reserve?\n")
	fmt.Fprintf(out, "  allocation only, nothing written; cap %d GiB, host-memory floor %d GiB\n",
		capGiB, bankHeadroomFloorGiB)
	fmt.Fprintf(out, "  type %d (heap %d, reported %.1f GiB): **%.1f GiB** — %s\n",
		t.Index, t.HeapIndex, float64(t.HeapSize)/(1<<30), float64(got)/(1<<30), why)
	fmt.Fprintf(out, "  run again with -bankheadroomtype to measure another type; "+
		"a second probe in this process would read zero (see the doc comment)\n")
	return nil
}

// fillWith appends bankBufferBytes allocations of memType to bufs until the
// driver refuses, the cap is hit, or host memory runs low.
func fillWith(dev *vk.Device, memType uint32, capBytes int, bufs *[]*vk.Buffer) (int, string) {
	got := 0
	for got < capBytes {
		if avail := memAvailableGiB(); avail > 0 && avail < bankHeadroomFloorGiB {
			return got, fmt.Sprintf("stopped at the floor, %.1f GiB of host memory left", avail)
		}
		b, err := dev.NewBufferOfType(bankBufferBytes, memType)
		if err != nil {
			return got, "the driver refused"
		}
		*bufs = append(*bufs, b)
		got += bankBufferBytes
	}
	return got, "hit the cap"
}

// memAvailableGiB reads MemAvailable, so the probe can stop before it starts
// costing the machine rather than after.
func memAvailableGiB() float64 {
	f, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(f), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		var kb float64
		if _, err := fmt.Sscanf(line, "MemAvailable: %f kB", &kb); err == nil {
			return kb / (1 << 20)
		}
	}
	return 0
}

// bankMaxParts bounds the workgroups-per-slab search, and so the dst buffer.
const bankMaxParts = 16

// bankParts splits a slab across workgroups. The grid needs to be wide
// enough to fill 40 CUs — one workgroup per 1.6 MB slab would launch 640 of
// them for a 1 GiB read, which is 16 per CU and fine, but at 4 MiB slabs it
// would be 256 and not fine. The constraint is that each part's chunk count
// stays a whole multiple of the workgroup width, so no lane is idle on the
// tail and every part reads the same bytes.
func bankParts(slabBytes int) (int, bool) {
	chunks := slabBytes / 16
	for parts := bankMaxParts; parts >= 1; parts /= 2 {
		if chunks%parts == 0 && (chunks/parts)%bankWG == 0 {
			return parts, true
		}
	}
	return 0, false
}

// bankDefaultType means "whatever vk.NewBuffer prefers", which on this device
// is always memory type 3 (DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT).
const bankDefaultType = ^uint32(0)

// allocBank allocates as much of want as the device will give, in
// bankBufferBytes pieces, dirties every page of it and builds one pipeline
// per piece. A partial bank is not an error: running out is itself
// information, and the sweep simply stops at what was obtained.
//
// memType is bankDefaultType for the preferred type or an explicit index for
// the L0b arm. An explicit type that is not host-visible comes back unmapped,
// and its pages are left however the driver supplied them — noted in the
// report, because an untouched allocation is the one way this probe can lie.
func allocBank(dev *vk.Device, mod *vk.ShaderModule, dst *vk.Buffer, want, maxSlabs int, memType uint32, out io.Writer) ([]bankBuf, error) {
	var bufs []bankBuf
	for off := 0; off < want; {
		size := bankBufferBytes
		if rem := want - off; rem < size {
			size = rem
		}
		size -= size % bankPageBytes
		if size == 0 {
			break
		}
		var src *vk.Buffer
		var err error
		if memType == bankDefaultType {
			src, err = dev.NewBuffer(size)
		} else {
			src, err = dev.NewBufferOfType(size, memType)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "bank: allocation stopped at %.1f GiB: %v\n",
				float64(off)/(1<<30), err)
			break
		}
		if src.Mapped() {
			touchPages(src)
		}
		// The slab table is always a default-type buffer: it has to be
		// writable from the host, and it is 32 KB, so where it lives cannot
		// affect a measurement that reads gigabytes.
		slabs, err := dev.NewBuffer(maxSlabs * 4)
		if err != nil {
			src.Destroy()
			return bufs, err
		}
		pipe, err := dev.NewPipeline(mod, vk.PipelineSpec{
			Buffers:          []*vk.Buffer{src, slabs, dst},
			PushConstantSize: 12,
		})
		if err != nil {
			src.Destroy()
			slabs.Destroy()
			return bufs, err
		}
		bufs = append(bufs, bankBuf{src: src, slabs: slabs, pipe: pipe, off: off, size: size})
		off += size
	}
	return bufs, nil
}

func (b bankBuf) release() {
	if b.pipe != nil {
		b.pipe.Destroy()
	}
	if b.slabs != nil {
		b.slabs.Destroy()
	}
	if b.src != nil {
		b.src.Destroy()
	}
}

// touchPages writes one word per page. Vulkan's contract is that the memory
// exists once vkAllocateMemory returns, but nothing promises a distinct
// physical page per virtual one until something dirties it, and a bank that
// is really one zero page repeated would read at cache speed and answer the
// wrong question. The value written is the page index, so a gross failure
// (every slab reading the same bytes) does not sum to the same total as a
// working one.
func touchPages(b *vk.Buffer) {
	base := b.MappedPointer()
	for off := 0; off < b.Size(); off += bankPageBytes {
		*(*uint32)(unsafe.Add(base, off)) = uint32(off / bankPageBytes)
	}
}

// bankSlotSpace is which slab-sized slots a prefix of the bank offers, and
// in which buffer each lives. Slots are numbered globally in address order,
// so an order can be expressed as a choice of global indices and mapped back
// afterwards.
type bankSlotSpace struct {
	total int
	// first[i] is the global index of buffer i's first slot; count[i] is how
	// many slots of it fall inside the prefix.
	first []int
	count []int
}

func bankSlots(bufs []bankBuf, prefix, slabBytes int) bankSlotSpace {
	s := bankSlotSpace{first: make([]int, len(bufs)), count: make([]int, len(bufs))}
	for i, b := range bufs {
		usable := prefix - b.off
		if usable > b.size {
			usable = b.size
		}
		if usable < 0 {
			usable = 0
		}
		s.first[i] = s.total
		s.count[i] = usable / slabBytes
		s.total += s.count[i]
	}
	return s
}

// locate maps a global slot index back to (buffer, slot within buffer).
func (s bankSlotSpace) locate(g int) (int, int) {
	i := sort.Search(len(s.first), func(i int) bool { return s.first[i]+s.count[i] > g }) //nolint:gocritic
	return i, g - s.first[i]
}

// selectSlabs picks n of the prefix's slots. Every order returns exactly n
// distinct slots, so the bytes read and the instruction count are identical
// across the three and only the addresses differ.
func selectSlabs(s bankSlotSpace, n int, order string) [][]int {
	global := make([]int, 0, n)
	switch order {
	case "seq":
		for j := 0; j < n; j++ {
			global = append(global, j)
		}
	case "spread":
		for j := 0; j < n; j++ {
			global = append(global, j*s.total/n)
		}
	case "rand":
		// Fixed seed: a cell has to be comparable with the same cell in the
		// next run, and TODO.md's handoffs quote a two-run agreement.
		r := rand.New(rand.NewSource(0x5EED))
		perm := r.Perm(s.total)[:n]
		sort.Ints(perm) // ascending, so only the *gaps* differ from spread
		global = perm
	}
	per := make([][]int, len(s.first))
	for _, g := range global {
		i, local := s.locate(g)
		per[i] = append(per[i], local)
	}
	return per
}

// loadSlabTables writes each buffer's chosen slots into its table and
// returns the dispatch list. Buffers with nothing selected are skipped
// rather than dispatched empty, so the dispatch count reported in Detail is
// how many of the bank's allocations the cell actually touched.
func loadSlabTables(bufs []bankBuf, pick [][]int, slabBytes, chunksPerPart, parts int) []vk.MultiDispatch {
	var out []vk.MultiDispatch
	dstBase := 0
	for i, slots := range pick {
		if len(slots) == 0 {
			continue
		}
		table := make([]uint32, len(slots))
		for j, local := range slots {
			table[j] = uint32(local * slabBytes / 16)
		}
		bufs[i].slabs.WriteBytes(uint32SliceToBytes(table))
		out = append(out, vk.MultiDispatch{
			Pipeline: bufs[i].pipe,
			GroupsX:  uint32(parts),
			GroupsY:  uint32(len(slots)),
			PushConstants: newPC().
				U32(uint32(chunksPerPart)).
				U32(uint32(parts)).
				U32(uint32(dstBase)).Bytes(),
		})
		dstBase += len(slots) * parts
	}
	return out
}

// verifyBankGather checks the kernel against a host sum over known bytes.
// The sweep itself cannot be verified — it reads tens of gigabytes of page
// indices — so correctness is established once, here, on a 16 MiB pattern
// written into the front of the first bank buffer. It catches the failures
// that would otherwise read as a *fast* result: a slab base computed wrong,
// a part reading the same chunks as its neighbour, a truncated loop.
func verifyBankGather(dev *vk.Device, b bankBuf, dst *vk.Buffer) error {
	const (
		slabBytes = 1 << 20
		slabs     = 16
		bytes     = slabBytes * slabs
	)
	parts, ok := bankParts(slabBytes)
	if !ok {
		return fmt.Errorf("verify: no legal part count for a %d B slab", slabBytes)
	}
	chunksPerPart := slabBytes / 16 / parts

	pattern := make([]byte, bytes)
	for i := range pattern {
		pattern[i] = byte(i*31 + i/1024)
	}
	b.src.WriteBytes(pattern)

	table := make([]uint32, slabs)
	for i := range table {
		table[i] = uint32(i * slabBytes / 16)
	}
	b.slabs.WriteBytes(uint32SliceToBytes(table))

	pc := newPC().U32(uint32(chunksPerPart)).U32(uint32(parts)).U32(0).Bytes()
	if _, err := vk.DispatchMultiTimed([]vk.MultiDispatch{{
		Pipeline: b.pipe, GroupsX: uint32(parts), GroupsY: slabs, PushConstants: pc,
	}}, 1, 1, false); err != nil {
		return fmt.Errorf("verify dispatch: %w", err)
	}

	// Each workgroup sums the uint32s of one part, in uint32 wraparound.
	want := make([]uint32, slabs*parts)
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&pattern[0])), bytes/4)
	for wg := range want {
		var s uint32
		base := wg * chunksPerPart * 4
		for i := 0; i < chunksPerPart*4; i++ {
			s += words[base+i]
		}
		want[wg] = s
	}
	got := dst.ReadUint32(len(want))
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("verify: workgroup %d summed %d, want %d", i, got[i], want[i])
		}
	}
	// Leave the pattern behind rather than restoring the page indices: these
	// pages are dirty either way, which is all the sweep needs of them.
	return nil
}

// PrintBankSummary prints one grid per slab size: prefix down the side,
// selection order across. The reading is in the columns — `seq` is the
// control and should not move, so any fall in `spread` and `rand` as the
// prefix grows is the working set, not the kernel.
func PrintBankSummary(w io.Writer, results []Result, p Params) {
	rows := filterOp(results, "bank")
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "\nL0a — does the DRAM bus survive a big weight bank?\n")
	fmt.Fprintf(w, "%d MiB read per step, GB/s, against the 242 GB/s a small-range W4A8 GEMV reaches (§1.7)\n",
		p.BankReadMiB)

	for _, slabBytes := range BankSlabBytesList {
		key := fmt.Sprintf("slabKB=%d;", slabBytes/1024)
		cells := map[[2]string]Result{}
		var gibs []int
		for _, r := range rows {
			if !strings.HasPrefix(r.Detail, key) {
				continue
			}
			k := [2]string{fmt.Sprint(r.Size), r.Variant}
			if _, seen := cells[k]; !seen {
				cells[k] = r
			}
			gibs = appendUnique(gibs, r.Size)
		}
		if len(gibs) == 0 {
			continue
		}
		sort.Ints(gibs)
		fmt.Fprintf(w, "\n  slab %d KB\n", slabBytes/1024)
		tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprint(tw, "  bank GiB")
		for _, o := range bankOrders {
			fmt.Fprintf(tw, "\t%s", o)
		}
		fmt.Fprint(tw, "\trand/seq\tsclk MHz\tW\n")
		for _, g := range gibs {
			fmt.Fprintf(tw, "  %d", g)
			var seq, rnd float64
			for _, o := range bankOrders {
				r, ok := cells[[2]string{fmt.Sprint(g), o}]
				if !ok {
					fmt.Fprint(tw, "\t-")
					continue
				}
				fmt.Fprintf(tw, "\t%.0f", r.GBPS)
				switch o {
				case "seq":
					seq = r.GBPS
				case "rand":
					rnd = r.GBPS
				}
			}
			ratio := "-"
			if seq > 0 && rnd > 0 {
				ratio = fmt.Sprintf("%.2fx", rnd/seq)
			}
			c := cells[[2]string{fmt.Sprint(g), "rand"}].Clocks
			fmt.Fprintf(tw, "\t%s\t%.0f\t%.0f\n", ratio, c.SclkMHz, c.PowerW)
		}
		tw.Flush()
	}
	PrintBankMemTypeSummary(w, results, p)
}

// PrintBankMemTypeSummary prints L0b: one row per memory type that can back a
// storage buffer, at two prefixes. §5.1 predicted a ranking worth a few
// percent; the column to read is whether there is any spread at all.
func PrintBankMemTypeSummary(w io.Writer, results []Result, p Params) {
	var rows []Result
	for _, r := range results {
		if r.Op == "bank" && r.Variant == "memtype" {
			rows = append(rows, r)
		}
	}
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "\nL0b / §5.1 — does the memory type matter for a GPU read?\n")
	fmt.Fprintf(w, "%d MiB read per step from a %d GiB bank of each type, GB/s\n",
		p.BankReadMiB, bankMemTypeGiB)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "  type\theap\tgot GiB")
	for _, g := range bankMemTypePrefixesGiB {
		fmt.Fprintf(tw, "\t%d GiB", g)
	}
	fmt.Fprint(tw, "\tflags\n")
	seen := map[string]bool{}
	for _, r := range rows {
		f := detailField(r.Detail, "type")
		if seen[f] {
			continue
		}
		seen[f] = true
		fmt.Fprintf(tw, "  %s\t%s\t%s", f, detailField(r.Detail, "heap"), detailField(r.Detail, "gotGiB"))
		for _, g := range bankMemTypePrefixesGiB {
			v := "-"
			for _, q := range rows {
				if detailField(q.Detail, "type") == f && q.Size == g {
					v = fmt.Sprintf("%.0f", q.GBPS)
				}
			}
			fmt.Fprintf(tw, "\t%s", v)
		}
		fmt.Fprintf(tw, "\t%s\n", detailField(r.Detail, "flags"))
	}
	tw.Flush()
}

func filterOp(results []Result, op string) []Result {
	var out []Result
	for _, r := range results {
		if r.Op == op {
			out = append(out, r)
		}
	}
	return out
}

func appendUnique(xs []int, x int) []int {
	for _, v := range xs {
		if v == x {
			return xs
		}
	}
	return append(xs, x)
}
