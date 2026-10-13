// Package audiobox runs Meta's Audiobox Aesthetics (facebook/
// audiobox-aesthetics, CC-BY 4.0) on the device (MUSIC.md A14c): four
// scores on a 1-10 scale for a piece of audio -- production quality (PQ),
// production complexity (PC), content enjoyment (CE) and content usefulness
// (CU) -- which ACE-Step 1.5's own report quotes for its songs.
//
// The model is a WavLM-base encoder and four small heads. Upstream's
// predictor (infer.AesPredictor) resamples to 16 kHz mono, cuts 10 s
// windows at a 10 s hop (the last zero-padded and masked), scores each, and
// averages the windows weighted by their length. Every score is the
// window's head output de-normalised with the checkpoint's mean and std.
// Upstream runs it in fp32 (its "bf16" precision never turns autocast on).
package audiobox

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

// The encoder's shape, which the device graph and the shaders compile in.
const (
	SampleRate    = 16000
	WindowSamples = 10 * SampleRate
	// Frames is a full window's frames after the 7-layer extractor (one per
	// 320 samples, less the kernels' edges).
	Frames = 499
	// framesPad is a window's rows in every arena: the attention's planes
	// are padded to 128 rows (the kokoro BERT's bertTokenAlign) and the
	// GEMM tiles divide it.
	framesPad = 512
	convDim   = 512
	hidden    = 768
	ffn       = 3072
	heads     = 12
	headDim   = 64
	layers    = 12
	posKernel = 128
	posGroups = 16
	// numBuckets and maxDistance are layer 0's relative-position buckets.
	numBuckets  = 320
	maxDistance = 800
	lnEps       = 1e-5
)

// convLayers are the extractor's (kernel, stride), 512 channels each.
var convLayers = [7][2]int{{10, 5}, {3, 2}, {3, 2}, {3, 2}, {3, 2}, {2, 2}, {2, 2}}

// convLens are each conv layer's output frames for a 10 s window.
var convLens = func() [7]int {
	var l [7]int
	n := WindowSamples
	for i, c := range convLayers {
		n = (n-c[0])/c[1] + 1
		l[i] = n
	}
	return l
}()

// Axes are the four scores, in the order Scores lists them.
var Axes = [4]string{"PQ", "PC", "CE", "CU"}

// Scores are one piece of audio's four aesthetics, each on upstream's 1-10
// scale (the heads are not clamped; a value can stray outside it).
type Scores struct {
	PQ, PC, CE, CU float64
}

// axis returns score a (in Axes' order).
func (s *Scores) axis(a int) *float64 {
	return [4]*float64{&s.PQ, &s.PC, &s.CE, &s.CU}[a]
}

// head is one axis: its softmax layer weights, the 5-layer MLP (four
// Linear-LayerNorm-GELU blocks and a Linear to one) and the target's
// de-normalisation.
type head struct {
	layerW    [layers + 1]float64 // softmax(layer_weights)
	w, b      [4][]float32        // [768][768] and [768]
	lnG, lnB  [4][]float32
	outW      []float32 // [768]
	outB      float32
	mean, std float64
}

// weights are the host's copy of what the device stages, plus the heads.
type weights struct {
	conv0          []float32    // [512][10]
	gnG, gnB       []float32    // GroupNorm(512, 512)
	conv           [6][]float32 // [512][512][k]
	feLNG, feLNB   []float32    // LayerNorm(512)
	projW, projB   []float32    // post_extract_proj [768][512]
	posW           []float32    // pos_conv, weight norm folded, [768][48][128]
	posB           []float32
	encLNG, encLNB []float32
	relBias        []float32 // layer 0's relative_attention_bias [320][12]
	layer          [layers]layerWeights
	heads          [4]head
}

type layerWeights struct {
	q, k, v, o, fc1, fc2       []float32
	qB, kB, vB, oB, fc1B, fc2B []float32
	ln1G, ln1B, ln2G, ln2B     []float32
	grepW, grepB, grepA        []float32 // [8][64], [8], [12]
}

// loadWeights reads model.safetensors.
func loadWeights(dir string) (*weights, error) {
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	var firstErr error
	get := func(name string, n int) []float32 {
		if firstErr != nil {
			return nil
		}
		t, err := set.Get(name)
		if err != nil {
			firstErr = err
			return nil
		}
		v, err := t.F32(nil)
		if err != nil {
			firstErr = err
			return nil
		}
		if n > 0 && len(v) != n {
			firstErr = fmt.Errorf("audiobox: %s has %d values, want %d", name, len(v), n)
		}
		return v
	}
	w := &weights{}
	p := "wavlm_model."
	w.conv0 = get(p+"feature_extractor.conv_layers.0.0.weight", convDim*10)
	w.gnG = get(p+"feature_extractor.conv_layers.0.2.weight", convDim)
	w.gnB = get(p+"feature_extractor.conv_layers.0.2.bias", convDim)
	for i := 1; i < 7; i++ {
		w.conv[i-1] = get(fmt.Sprintf("%sfeature_extractor.conv_layers.%d.0.weight", p, i), convDim*convDim*convLayers[i][0])
	}
	w.feLNG = get(p+"layer_norm.weight", convDim)
	w.feLNB = get(p+"layer_norm.bias", convDim)
	w.projW = get(p+"post_extract_proj.weight", hidden*convDim)
	w.projB = get(p+"post_extract_proj.bias", hidden)
	v := get(p+"encoder.pos_conv.0.weight_v", hidden*hidden/posGroups*posKernel)
	g := get(p+"encoder.pos_conv.0.weight_g", posKernel)
	w.posB = get(p+"encoder.pos_conv.0.bias", hidden)
	w.encLNG = get(p+"encoder.layer_norm.weight", hidden)
	w.encLNB = get(p+"encoder.layer_norm.bias", hidden)
	w.relBias = get(p+"encoder.layers.0.self_attn.relative_attention_bias.weight", numBuckets*heads)
	for l := range w.layer {
		lp := fmt.Sprintf("%sencoder.layers.%d.", p, l)
		L := &w.layer[l]
		L.q, L.qB = get(lp+"self_attn.q_proj.weight", hidden*hidden), get(lp+"self_attn.q_proj.bias", hidden)
		L.k, L.kB = get(lp+"self_attn.k_proj.weight", hidden*hidden), get(lp+"self_attn.k_proj.bias", hidden)
		L.v, L.vB = get(lp+"self_attn.v_proj.weight", hidden*hidden), get(lp+"self_attn.v_proj.bias", hidden)
		L.o, L.oB = get(lp+"self_attn.out_proj.weight", hidden*hidden), get(lp+"self_attn.out_proj.bias", hidden)
		L.fc1, L.fc1B = get(lp+"fc1.weight", ffn*hidden), get(lp+"fc1.bias", ffn)
		L.fc2, L.fc2B = get(lp+"fc2.weight", hidden*ffn), get(lp+"fc2.bias", hidden)
		L.ln1G, L.ln1B = get(lp+"self_attn_layer_norm.weight", hidden), get(lp+"self_attn_layer_norm.bias", hidden)
		L.ln2G, L.ln2B = get(lp+"final_layer_norm.weight", hidden), get(lp+"final_layer_norm.bias", hidden)
		L.grepW, L.grepB = get(lp+"self_attn.grep_linear.weight", 8*headDim), get(lp+"self_attn.grep_linear.bias", 8)
		L.grepA = get(lp+"self_attn.grep_a", heads)
	}
	targets, err := loadTargets(dir)
	if err != nil {
		return nil, err
	}
	for a, name := range Axes {
		h := &w.heads[a]
		lw := get("layer_weights."+name, layers+1)
		if firstErr != nil {
			break
		}
		softmax64(lw, h.layerW[:])
		for i := 0; i < 4; i++ {
			// proj_layer.<axis>: Linear at 0, 3, 6, 9; LayerNorm at 1, 4, 7, 10.
			h.w[i] = get(fmt.Sprintf("proj_layer.%s.%d.weight", name, 3*i), hidden*hidden)
			h.b[i] = get(fmt.Sprintf("proj_layer.%s.%d.bias", name, 3*i), hidden)
			h.lnG[i] = get(fmt.Sprintf("proj_layer.%s.%d.weight", name, 3*i+1), hidden)
			h.lnB[i] = get(fmt.Sprintf("proj_layer.%s.%d.bias", name, 3*i+1), hidden)
		}
		h.outW = get("proj_layer."+name+".12.weight", hidden)
		if ob := get("proj_layer."+name+".12.bias", 1); ob != nil {
			h.outB = ob[0]
		}
		h.mean, h.std = targets[name][0], targets[name][1]
	}
	if firstErr != nil {
		return nil, firstErr
	}
	w.posW = foldWeightNorm(v, g)
	return w, nil
}

// foldWeightNorm is torch's _weight_norm at dim 2 over pos_conv's
// [768][48][128]: each kernel position k scaled by g[k] over the norm of
// v[:, :, k], in fp32 as torch computes it (the sum of squares accumulated
// wider here).
func foldWeightNorm(v, g []float32) []float32 {
	cin := hidden / posGroups
	out := make([]float32, len(v))
	for k := 0; k < posKernel; k++ {
		var s float64
		for o := 0; o < hidden; o++ {
			for j := 0; j < cin; j++ {
				x := float64(v[(o*cin+j)*posKernel+k])
				s += x * x
			}
		}
		f := g[k] / float32(math.Sqrt(s))
		for o := 0; o < hidden; o++ {
			for j := 0; j < cin; j++ {
				i := (o*cin+j)*posKernel + k
				out[i] = v[i] * f
			}
		}
	}
	return out
}

// bucket is MultiheadAttention._relative_positions_bucket for one offset
// (key − query), bidirectional: 160 buckets a side, exact below 80, then
// logarithmic out to 800, in torch's fp32.
func bucket(d int) int {
	nb := numBuckets / 2
	b := 0
	if d > 0 {
		b = nb
	}
	a := d
	if a < 0 {
		a = -a
	}
	maxExact := nb / 2
	if a < maxExact {
		return b + a
	}
	r := float32(math.Log(float64(float32(a) / float32(maxExact))))
	r = r / float32(math.Log(float64(maxDistance)/float64(maxExact))) * float32(nb-maxExact)
	return b + min(maxExact+int(r), nb-1)
}

func softmax64(x []float32, out []float64) {
	mx := math.Inf(-1)
	for _, v := range x {
		mx = math.Max(mx, float64(v))
	}
	var s float64
	for i, v := range x {
		out[i] = math.Exp(float64(v) - mx)
		s += out[i]
	}
	for i := range x {
		out[i] /= s
	}
}

// parallel splits [0, n) over the cores.
func parallel(n int, fn func(lo, hi int)) {
	w := min(runtime.GOMAXPROCS(0), n)
	if w <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, min(lo+chunk, n))
	}
	wg.Wait()
}
