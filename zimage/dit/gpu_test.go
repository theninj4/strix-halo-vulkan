package dit

import (
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
	dev, err := vk.NewDevice(phys, qf, vk.DeviceFeatures{})
	if err != nil {
		inst.Destroy()
		t.Skipf("no device: %v", err)
	}
	t.Logf("device: %s", phys.Name)
	return dev, func() { dev.Destroy(); inst.Destroy() }
}

// TestGPUAttention checks the Vulkan attention stack against the same
// diffusers reference the CPU implementation is held to, in two pieces.
//
// The first feeds q/k/v already normed and rotated, so a failure is the
// attention kernel alone. The second feeds the raw projections and lets the
// GPU do the per-head RMS norms and the rotary embedding too, so those
// kernels are covered by the same comparison.
func TestGPUAttention(t *testing.T) {
	if _, err := os.Stat(transformer); err != nil {
		t.Skipf("no transformer checkpoint at %s", transformer)
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	blk, err := LoadBlock(set, "layers.0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	rope := ropeFromRef(t, m)

	dev, done := newTestDevice(t)
	defer done()

	g, err := NewGPUAttention(dev, blk.Attn, rope, m.Seq)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	wantCtx := loadRef(t, m, "attn_ctx")

	t.Run("attention only", func(t *testing.T) {
		got, err := g.Apply(
			loadRef(t, m, "q_roped").Clone(),
			loadRef(t, m, "k_roped").Clone(),
			loadRef(t, m, "v").Clone(),
			false,
		)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "gpu attn_ctx", got, wantCtx)
	})

	t.Run("with qk norm and rope", func(t *testing.T) {
		// The raw projections: the GPU has to reproduce the per-head norms
		// and the rotary embedding to land on the same context.
		attnIn := loadRef(t, m, "attn_in")
		q, err := blk.Attn.Q.Apply(attnIn)
		if err != nil {
			t.Fatal(err)
		}
		k, err := blk.Attn.K.Apply(attnIn)
		if err != nil {
			t.Fatal(err)
		}
		v, err := blk.Attn.V.Apply(attnIn)
		if err != nil {
			t.Fatal(err)
		}
		got, err := g.Apply(q, k, v, true)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "gpu attn_ctx (full)", got, wantCtx)
	})
}

// TestGPUMatchesCPU compares the two implementations directly, which stays
// meaningful once the GPU path moves to fp16 and the diffusers tolerance has
// to widen.
func TestGPUMatchesCPU(t *testing.T) {
	if _, err := os.Stat(transformer); err != nil {
		t.Skipf("no transformer checkpoint at %s", transformer)
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	blk, err := LoadBlock(set, "layers.0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	rope := ropeFromRef(t, m)

	dev, done := newTestDevice(t)
	defer done()
	g, err := NewGPUAttention(dev, blk.Attn, rope, m.Seq)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	q := loadRef(t, m, "q_roped")
	k := loadRef(t, m, "k_roped")
	v := loadRef(t, m, "v")

	gotGPU, err := g.Apply(q.Clone(), k.Clone(), v.Clone(), false)
	if err != nil {
		t.Fatal(err)
	}
	gotCPU, err := blk.Attn.scores(q, k, v)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "gpu vs cpu", gotGPU, gotCPU)
}

// TestAttentionKernelsAgree checks the tiled kernel against the simple one
// on the same inputs. They share no code beyond the transpose, and the
// simple one is small enough to read in one sitting, so it is the more
// trustworthy of the two -- which makes this the check that the online
// softmax's running max and rescaling are right, as opposed to merely
// self-consistent.
func TestAttentionKernelsAgree(t *testing.T) {
	if _, err := os.Stat(transformer); err != nil {
		t.Skipf("no transformer checkpoint at %s", transformer)
	}
	m := loadManifest(t)
	cfg, err := LoadConfig(transformer)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenSet(transformer)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	blk, err := LoadBlock(set, "layers.0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	rope := ropeFromRef(t, m)

	dev, done := newTestDevice(t)
	defer done()

	q := loadRef(t, m, "q_roped")
	k := loadRef(t, m, "k_roped")
	v := loadRef(t, m, "v")

	run := func(flash bool) *Mat {
		g, err := NewGPUAttention(dev, blk.Attn, rope, m.Seq)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()
		g.Flash = flash
		out, err := g.Apply(q.Clone(), k.Clone(), v.Clone(), false)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	compare(t, "flash vs simple", run(true), run(false))
}
