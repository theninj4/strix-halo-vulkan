package dit

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const strixHaloDeviceID = 0x1586

func newTestDevice(t *testing.T) (*vk.Device, func()) {
	t.Helper()
	inst, err := vk.NewInstance("dit-test")
	if err != nil {
		t.Skipf("no Vulkan instance: %v", err)
	}
	devices, err := inst.PhysicalDevices()
	if err != nil || len(devices) == 0 {
		inst.Destroy()
		t.Skipf("no Vulkan devices: %v", err)
	}
	phys := &devices[0]
	for i := range devices {
		if devices[i].DeviceID == strixHaloDeviceID {
			phys = &devices[i]
			break
		}
	}
	qf, err := phys.ComputeQueueFamily()
	if err != nil {
		inst.Destroy()
		t.Skipf("no compute queue: %v", err)
	}
	// fp16 storage and arithmetic plus cooperative matrix are what the
	// stage-3c kernels need, and subgroup size control is what lets a wave32
	// variant be pinned to the size its SPIR-V was built for (IDEAS §6.2). A
	// device missing any of them still runs the scalar kernels: NewGPUAttention
	// builds the WMMA ladder only where it can.
	sgs, err := phys.SubgroupSizeControl()
	if err != nil {
		inst.Destroy()
		t.Skipf("no subgroup size control query: %v", err)
	}
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{
		Float16: true, CoopMatrix: true, SubgroupSizeControl: sgs.Supported,
	})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// fp16RelTol is the bound the matrix-core kernels are held to, set the same
// way every other tolerance in this project was: from the measured drift, with
// a negative control pinning the other side.
//
// The scalar kernels are fp32 throughout and stay inside the 2e-4 every stage
// shares -- 1.1e-5 measured. The WMMA path narrows q, k, v and the softmax
// weights to fp16, which is not a choice but the only operand type this
// device's matrix cores take, and every variant of it measures **6.7e-3**
// against the diffusers reference at 320 tokens. The bound is 1.5e-2, a little
// over twice that.
//
// That figure is what fp16 predicts once the normalisation is accounted for.
// The worst element is 0.031 off a value of 52, i.e. 6e-4 relative to itself,
// which is fp16's 4.9e-4 quantum; it looks eleven times larger here only
// because this project normalises by the tensor's RMS (4.73) rather than per
// element, deliberately, since these activations cross zero constantly. The
// eight variants agree with each other to the last digit of that, which is the
// other half of the evidence: the error follows the arithmetic, not the tiling.
//
// Worth being explicit about what this is not. It is not a claim that fp16
// attention is as accurate as fp32 attention. It is a claim that the error is
// the one fp16 rounding predicts rather than a bug, and that it sits two to
// three orders of magnitude below every way of getting this kernel wrong that
// has been tried (TestGPUWMMANegativeControls). Stage 2 found fp16 could not
// hold the VAE's intermediates at all -- scores of 1.16e7 against a 65504
// limit -- and the reason the same narrowing is safe here is structural: the
// softmax weights are bounded by 1 by construction and every accumulator stays
// fp32.
const fp16RelTol = 1.5e-2

// tolFor is the bound a given kernel is held to.
func tolFor(k Kernel) float64 {
	if k == KernelSimple || k == KernelFlash {
		return relTol
	}
	return fp16RelTol
}

// fixture is the reference block, its rotary table and the open checkpoint.
type fixture struct {
	m     *manifest
	cfg   *Config
	blk   *Block
	rope  *RoPE
	close func()
}

func loadFixture(t *testing.T) fixture {
	t.Helper()
	if _, err := os.Stat(transformer); err != nil {
		t.Skipf("no transformer checkpoint at %s", transformer)
	}
	m := loadManifest(t, refDir)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	blk, err := LoadBlock(set, "layers.0", cfg)
	if err != nil {
		set.Close()
		t.Fatal(err)
	}
	return fixture{m: m, cfg: cfg, blk: blk, rope: ropeFromRef(t, m), close: func() { set.Close() }}
}

// TestGPUAttention checks the Vulkan attention stack against the same
// diffusers reference the CPU implementation is held to, in two pieces, for
// every kernel the device can run.
//
// The first feeds q/k/v already normed and rotated, so a failure is the
// attention kernel alone. The second feeds the raw projections and lets the
// GPU do the per-head RMS norms and the rotary embedding too, so those
// kernels are covered by the same comparison.
func TestGPUAttention(t *testing.T) {
	f := loadFixture(t)
	defer f.close()

	dev, done := newTestDevice(t)
	defer done()

	g, err := NewGPUAttention(dev, f.blk.Attn, f.rope, f.m.Seq)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	t.Logf("kernels: %v (default %s)", g.Kernels(), g.Kernel)

	wantCtx := loadRef(t, f.m, "attn_ctx")
	attnIn := loadRef(t, f.m, "attn_in")

	for _, kernel := range g.Kernels() {
		g.Kernel = kernel
		t.Run(string(kernel), func(t *testing.T) {
			got, err := g.Apply(
				loadRef(t, f.m, "q_roped").Clone(),
				loadRef(t, f.m, "k_roped").Clone(),
				loadRef(t, f.m, "v").Clone(),
				false,
			)
			if err != nil {
				t.Fatal(err)
			}
			compareTol(t, "attn_ctx", got, wantCtx, tolFor(kernel))

			// The raw projections: the GPU has to reproduce the per-head norms
			// and the rotary embedding to land on the same context.
			q, err := f.blk.Attn.Q.Apply(attnIn)
			if err != nil {
				t.Fatal(err)
			}
			k, err := f.blk.Attn.K.Apply(attnIn)
			if err != nil {
				t.Fatal(err)
			}
			v, err := f.blk.Attn.V.Apply(attnIn)
			if err != nil {
				t.Fatal(err)
			}
			got, err = g.Apply(q, k, v, true)
			if err != nil {
				t.Fatal(err)
			}
			compareTol(t, "attn_ctx (full)", got, wantCtx, tolFor(kernel))
		})
	}
}

// TestGPUMatchesCPU compares each GPU kernel against the CPU implementation
// directly. It is the comparison that stays meaningful now the matrix-core
// path is fp16 and its tolerance against diffusers has had to widen: the CPU
// reference and the fp32 kernels agree to 2e-4, so anything the fp16 path adds
// shows up here as its own number rather than hiding inside one shared bound.
func TestGPUMatchesCPU(t *testing.T) {
	f := loadFixture(t)
	defer f.close()

	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUAttention(dev, f.blk.Attn, f.rope, f.m.Seq)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	q := loadRef(t, f.m, "q_roped")
	k := loadRef(t, f.m, "k_roped")
	v := loadRef(t, f.m, "v")
	gotCPU, err := f.blk.Attn.scores(q, k, v)
	if err != nil {
		t.Fatal(err)
	}

	for _, kernel := range g.Kernels() {
		g.Kernel = kernel
		gotGPU, err := g.Apply(q.Clone(), k.Clone(), v.Clone(), false)
		if err != nil {
			t.Fatalf("%s: %v", kernel, err)
		}
		compareTol(t, string(kernel)+" vs cpu", gotGPU, gotCPU, tolFor(kernel))
	}
}

// TestAttentionKernelsAgree checks every kernel against the simple one on the
// same inputs. They share no code -- different shaders, different operand
// layouts, fp32 against fp16 -- and the simple one is small enough to read in
// one sitting, so it is the most trustworthy of them. This is therefore the
// check that each kernel's online softmax, its running max and its rescaling
// are right, as opposed to merely self-consistent.
func TestAttentionKernelsAgree(t *testing.T) {
	f := loadFixture(t)
	defer f.close()

	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUAttention(dev, f.blk.Attn, f.rope, f.m.Seq)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	q := loadRef(t, f.m, "q_roped")
	k := loadRef(t, f.m, "k_roped")
	v := loadRef(t, f.m, "v")

	run := func(kernel Kernel) *Mat {
		g.Kernel = kernel
		out, err := g.Apply(q.Clone(), k.Clone(), v.Clone(), false)
		if err != nil {
			t.Fatalf("%s: %v", kernel, err)
		}
		return out
	}
	want := run(KernelSimple)
	for _, kernel := range g.Kernels()[1:] {
		compareTol(t, string(kernel)+" vs simple", run(kernel), want, tolFor(kernel))
	}
}

// randomMats builds q, k and v for a sequence length the reference dump does
// not cover, so the tail paths can be exercised at a length that is not a
// multiple of anybody's tile.
func randomMats(seed int64, tokens, dim int) (*Mat, *Mat, *Mat) {
	rng := rand.New(rand.NewSource(seed))
	mk := func() *Mat {
		m := NewMat(tokens, dim)
		for i := range m.Data {
			m.Data[i] = float32(rng.NormFloat64())
		}
		return m
	}
	return mk(), mk(), mk()
}

// TestGPUWMMATails runs the matrix-core kernels at sequence lengths that are
// not multiples of their query block or their key block, which is the case
// every mask in the kernel exists for. 4096 tokens divides by everything; the
// real model also runs a 320-token caption stream, and nothing guarantees the
// next stage's lengths are round either.
func TestGPUWMMATails(t *testing.T) {
	f := loadFixture(t)
	defer f.close()

	dev, done := newTestDevice(t)
	defer done()

	for _, tokens := range []int{300, 337, 512} {
		q, k, v := randomMats(3, tokens, f.cfg.Dim)
		ids := make([][3]int32, tokens)
		for i := range ids {
			ids[i] = [3]int32{int32(i), 0, 0}
		}
		rope, err := NewRoPE(ids, f.cfg.AxesDims, f.cfg.AxesLens, f.cfg.RopeTheta)
		if err != nil {
			t.Fatal(err)
		}
		g, err := NewGPUAttention(dev, f.blk.Attn, rope, tokens)
		if err != nil {
			t.Fatal(err)
		}
		gotCPU, err := f.blk.Attn.scores(q, k, v)
		if err != nil {
			t.Fatal(err)
		}
		for _, kernel := range g.Kernels() {
			g.Kernel = kernel
			got, err := g.Apply(q.Clone(), k.Clone(), v.Clone(), false)
			if err != nil {
				t.Fatalf("%s at %d tokens: %v", kernel, tokens, err)
			}
			compareTol(t, fmt.Sprintf("%s @%d", kernel, tokens), got, gotCPU, tolFor(kernel))
		}
		g.Destroy()
	}
}

// TestGPUWMMANegativeControls breaks the matrix-core path in the three ways it
// was actually at risk of being broken, and asserts each break is caught. A
// test without one of these proves only that the kernel runs.
//
// Each control is a single change to an otherwise identical dispatch:
//
//   - **v packed in the natural layout.** The two GEMMs reduce over different
//     axes -- q.k^T over headDim, p.v over the token -- and the hardware
//     operand layout is K-contiguous for both operands, so q and k want the
//     natural per-head layout and v wants its transpose. Getting this wrong is
//     the same class of mistake the scalar kernel made in the other direction
//     (dit_transpose_k.comp), and it is silent: the shapes still match and
//     every fragment load is in bounds.
//   - **q without log2(e).** The softmax's exponential is exp2, which is one
//     instruction where exp is two, and the scale that makes that legal is
//     folded into q by the pack rather than applied in the kernel. The two
//     halves of that trick live in different files, so this control is what
//     keeps them together.
//   - **the tail mask compiled out** (wmma_qt2_kt4_nomask, built from the same
//     source with -DNO_TAIL_MASK). A pad key scores zero, not minus infinity,
//     so without the mask every row is divided by a denominator counting keys
//     that do not exist.
func TestGPUWMMANegativeControls(t *testing.T) {
	f := loadFixture(t)
	defer f.close()

	dev, done := newTestDevice(t)
	defer done()

	// 300 tokens: not a multiple of any variant's key block, so the tail mask
	// is live and the third control has something to get wrong.
	const tokens = 300
	q, k, v := randomMats(11, tokens, f.cfg.Dim)
	ids := make([][3]int32, tokens)
	for i := range ids {
		ids[i] = [3]int32{int32(i), 0, 0}
	}
	rope, err := NewRoPE(ids, f.cfg.AxesDims, f.cfg.AxesLens, f.cfg.RopeTheta)
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGPUAttention(dev, f.blk.Attn, rope, tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	if _, ok := g.wmma[DefaultKernel]; !ok {
		t.Skip("device has no matrix-core kernels")
	}
	want, err := f.blk.Attn.scores(q, k, v)
	if err != nil {
		t.Fatal(err)
	}

	run := func(kernel Kernel, ctl controls) float64 {
		g.Kernel, g.ctl = kernel, ctl
		got, err := g.Apply(q.Clone(), k.Clone(), v.Clone(), false)
		if err != nil {
			t.Fatalf("%s: %v", kernel, err)
		}
		_, _, rel, _ := deviation(got, want)
		return rel
	}

	base := run(DefaultKernel, controls{})
	if base > fp16RelTol {
		t.Fatalf("unbroken kernel is already outside the bound: %.3g", base)
	}
	for _, c := range []struct {
		name   string
		kernel Kernel
		ctl    controls
	}{
		{"v in the natural layout", DefaultKernel, controls{vNatural: true}},
		{"q without log2(e)", DefaultKernel, controls{noLog2E: true}},
		{"tail mask compiled out", wmmaControl.name, controls{}},
	} {
		rel := run(c.kernel, c.ctl)
		if rel <= fp16RelTol {
			t.Errorf("%s: rel %.3g is inside the %.0e bound -- the test cannot see this break",
				c.name, rel, fp16RelTol)
			continue
		}
		t.Logf("%-24s rel %8.3g = %6.0fx the bound", c.name, rel, rel/fp16RelTol)
	}
	g.ctl = controls{}
}
