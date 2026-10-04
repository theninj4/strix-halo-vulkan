package gemma4

import (
	"math"
	"math/rand/v2"
	"testing"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/shaders"
	"strix-halo-vulkan/vk"
)

// TestAttentionKinds is R5's attention gate on synthetic data: the prep and
// attention kernels of both layer kinds over one packed pass laid out the
// way Kev's planner lays one out (a 1,100-token state, then two branches at
// 64-aligned cells), against a float64 reference with HF's semantics: scaled
// q/k norms, unscaled v norm (V = raw k on the full layers), NeoX rope from
// the same table, scale 1.0, causal, and on the sliding layers a 1,024-token
// window **on positions**. The state is longer than the window, so the
// branches' view of the state's start is cut by it.
func TestAttentionKinds(t *testing.T) {
	if testing.Short() {
		t.Skip("GPU")
	}
	dev, done := newTestDevice(t)
	defer done()

	const S, A, B = 1100, 40, 30
	cellA, cellB := 1152, 1216 // roundUp(S, 64), roundUp(cellA+A, 64)
	n := S + A + B
	cells := cellB + B + 64
	type row struct{ pos, cell, segStart int }
	rows := make([]row, 0, n)
	meta := make([]uint32, 0, 8*n)
	for i := range S {
		rows = append(rows, row{i, i, 0})
		meta = append(meta, uint32(i), 0, 0, uint32(i), 0, 0, 0, S)
	}
	for i := range A {
		rows = append(rows, row{S + i, cellA + i, S})
		meta = append(meta, uint32(S+i), 1, S, uint32(cellA+i), 0, S, 0, S)
	}
	for i := range B {
		rows = append(rows, row{S + i, cellB + i, S + A})
		meta = append(meta, uint32(S+i), 1, S+A, uint32(cellB+i), 0, S, 0, S)
	}

	for _, kind := range []struct {
		name         string
		hd, nkv, win int
		keqv         bool
		theta        float64
		rotated      int // frequencies that rotate; the rest are zero
		prep, attn   []byte
	}{
		{"sliding", 256, 8, 1024, false, 1e4, 128, shaders.Gemma4AttnPrepSlide, shaders.Gemma4AttnSlide},
		{"full", 512, 2, 0, true, 1e6, 64, shaders.Gemma4AttnPrepFull, shaders.Gemma4AttnFull},
		{"full-gqa", 512, 2, 0, true, 1e6, 64, shaders.Gemma4AttnPrepFull, shaders.Gemma4AttnGQA},
	} {
		t.Run(kind.name, func(t *testing.T) {
			hd, nkv, half := kind.hd, kind.nkv, kind.hd/2
			const nq = 16
			width := nq*hd + nkv*hd
			if !kind.keqv {
				width += nkv * hd
			}
			r := rand.New(rand.NewPCG(3, uint64(hd)))
			// Weights arena: q_norm, k_norm, then the cos and sin tables.
			span := S + A
			w := make([]float32, 2*hd+2*span*half)
			for i := range 2 * hd {
				w[i] = float32(0.25 + 0.05*r.NormFloat64())
			}
			cosT, sinT := w[2*hd:2*hd+span*half], w[2*hd+span*half:]
			for p := range span {
				for j := range half {
					inv := float32(0)
					if j < kind.rotated {
						inv = float32(1 / math.Pow(kind.theta, float64(2*j)/float64(hd)))
					}
					a := float64(float32(p) * inv)
					cosT[p*half+j], sinT[p*half+j] = float32(math.Cos(a)), float32(math.Sin(a))
				}
			}
			// Activation arena: P then the metadata.
			P := make([]float32, n*width)
			for i := range P {
				P[i] = float32(r.NormFloat64())
			}
			aMeta := len(P)
			// fp16 arena: q, K, V, out.
			lda := nq*hd + 128
			hQ, hK := 0, n*nq*hd
			hV := hK + cells*nkv*hd
			hO := hV + cells*nkv*hd

			wbuf, _ := dev.NewBuffer(len(w) * 4)
			abuf, _ := dev.NewBuffer((len(P) + len(meta)) * 4)
			hbuf, _ := dev.NewBuffer((hO + n*lda) * 2)
			bank, _ := dev.NewBuffer(256)
			defer wbuf.Destroy()
			defer abuf.Destroy()
			defer hbuf.Destroy()
			defer bank.Destroy()
			wbuf.WriteFloat32(w)
			abuf.WriteFloat32(P)
			abuf.WriteUint32At(aMeta, meta)
			hbuf.Zero()

			spec := vk.PipelineSpec{Buffers: []*vk.Buffer{wbuf, abuf, hbuf, bank}, PushConstantSize: pushBytes, RequiredSubgroupSize: 64}
			var pipes []*vk.ComputePipeline
			for _, spv := range [][]byte{kind.prep, kind.attn} {
				mod, err := dev.NewShaderModule(spv)
				if err != nil {
					t.Fatal(err)
				}
				defer mod.Destroy()
				s := spec
				if len(pipes) == 0 {
					s.RequiredSubgroupSize = 0 // the prep is one thread a dimension
				}
				pipe, err := dev.NewPipeline(mod, s)
				if err != nil {
					t.Fatal(err)
				}
				defer pipe.Destroy()
				pipes = append(pipes, pipe)
			}
			prep := push{InOff: 0, Aux0: uint32(width), BOff: uint32(hQ), KOff: uint32(hK), VOff: uint32(hV),
				WOff: 0, Aux1: uint32(aMeta), Aux2: uint32(2 * hd), Span: uint32(span),
				Eps: math.Float32bits(1e-6), Tokens: uint32(n)}
			attn := push{InOff: uint32(hQ), KOff: uint32(hK), VOff: uint32(hV), OutOff: uint32(hO),
				LDA: uint32(lda), Aux1: uint32(aMeta), Tokens: uint32(n)}
			d := []vk.MultiDispatch{
				{Pipeline: pipes[0], GroupsX: uint32(n), GroupsY: 1, PushConstants: prep.bytes()},
				{Pipeline: pipes[1], GroupsX: uint32((n + 15) / 16), GroupsY: attnGroupsY(kind.name, nq, hd, nkv), PushConstants: attn.bytes()},
			}
			ms, err := vk.DispatchMultiTimed(d, 1, 1, true)
			if err != nil {
				t.Fatal(err)
			}
			got := hbuf.ReadUint16At(hO, n*lda)

			// The reference, float64.
			norm := func(x []float64, wt []float32) []float64 {
				ss := 0.0
				for _, v := range x {
					ss += v * v
				}
				inv := 1 / math.Sqrt(ss/float64(len(x))+1e-6)
				y := make([]float64, len(x))
				for i, v := range x {
					y[i] = v * inv
					if wt != nil {
						y[i] *= float64(wt[i])
					}
				}
				return y
			}
			rope := func(x []float64, pos int) []float64 {
				y := make([]float64, hd)
				for j := range half {
					c, s := float64(cosT[pos*half+j]), float64(sinT[pos*half+j])
					y[j] = x[j]*c - x[j+half]*s
					y[j+half] = x[j+half]*c + x[j]*s
				}
				return y
			}
			vec := func(rw, off int) []float64 {
				v := make([]float64, hd)
				for i := range v {
					v[i] = float64(P[rw*width+off+i])
				}
				return v
			}
			Q := make([][][]float64, n)
			K := make([][][]float64, n)
			V := make([][][]float64, n)
			for i := range n {
				for h := range nq {
					Q[i] = append(Q[i], rope(norm(vec(i, h*hd), w[:hd]), rows[i].pos))
				}
				for g := range nkv {
					kr := vec(i, nq*hd+g*hd)
					K[i] = append(K[i], rope(norm(kr, w[hd:2*hd]), rows[i].pos))
					vr := kr
					if !kind.keqv {
						vr = vec(i, nq*hd+nkv*hd+g*hd)
					}
					V[i] = append(V[i], norm(vr, nil))
				}
			}
			var worst float64
			for i := range n {
				// The keys row i may see: its request's state (for a branch)
				// and its own segment up to itself, within the window.
				var keys []int
				if i >= S {
					for k := range S {
						keys = append(keys, k)
					}
				}
				for k := rows[i].segStart; k <= i; k++ {
					keys = append(keys, k)
				}
				for h := range nq {
					g := h / (nq / nkv)
					var sc []float64
					var ks []int
					for _, k := range keys {
						if kind.win > 0 && rows[i].pos-rows[k].pos >= kind.win {
							continue
						}
						s := 0.0
						for j := range hd {
							s += Q[i][h][j] * K[k][g][j]
						}
						sc, ks = append(sc, s), append(ks, k)
					}
					mx := math.Inf(-1)
					for _, s := range sc {
						mx = max(mx, s)
					}
					den := 0.0
					for _, s := range sc {
						den += math.Exp(s - mx)
					}
					for j := range hd {
						o := 0.0
						for a, k := range ks {
							o += math.Exp(sc[a]-mx) / den * V[k][g][j]
						}
						gv := float64(f16bits(got[i*lda+h*hd+j]))
						worst = max(worst, math.Abs(gv-o))
					}
				}
			}
			t.Logf("%d rows, %v; worst |o - ref| %.2e", n, ms, worst)
			if worst > 2e-2 {
				t.Errorf("worst error %.3e", worst)
			}
		})
	}
}

func f16bits(h uint16) float32 { return safetensors.F16ToF32(h) }

// attnGroupsY is the attention grid's y: head and half for gemma4_attn,
// KV head and half for gemma4_attn_gqa.
func attnGroupsY(name string, nq, hd, nkv int) uint32 {
	if name == "full-gqa" {
		return uint32(2 * nkv)
	}
	return uint32(nq * hd / 256)
}
