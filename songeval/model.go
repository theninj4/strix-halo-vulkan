// Package songeval runs SongEval (ASLP-lab, CC BY-NC-SA 4.0) on the device
// (MUSIC.md A14d): five whole-song aesthetic scores on a 1-5 scale --
// coherence, musicality, memorability, clarity of structure, naturalness of
// vocal phrasing -- from a small head over MuQ-large-msd-iter's (OpenMuQ,
// CC-BY-NC 4.0) layer-6 hidden states.
//
// SongEval's eval.py loads the song at 24 kHz mono, runs MuQ over all of it
// in one pass (a 128-band log-mel at 100 Hz, a Res2d conv stack down to
// 25 Hz, then a wav2vec2-conformer: rotary attention, a depthwise-conv
// module, half-step FFNs), takes hidden_states[6] -- the sixth layer's
// output -- and runs the head: an FFN, four 8-head self-attention layers
// chained without residuals, the mean of the last over time beside the max
// of the FFN's, a linear to five, tanh·2 + 3. MuQ wants fp32 ("NaN
// otherwise", its README); here every GEMM operand is fp16 with fp32 sums,
// the residual stream fp32, priced against the oracle at A14d.
package songeval

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"strix-halo-vulkan/safetensors"
)

const (
	SampleRate = 24000
	hop        = 240
	nFFT       = 2048
	nMels      = 128
	convDim    = 512
	hidden     = 1024
	ffn        = 4096
	heads      = 16
	headDim    = 64
	dwKernel   = 31
	// Layers is how many conformer layers SongEval reads: hidden_states[6].
	Layers = 6
	// head
	headHeads = 8
	headDim2  = 128
	lnEps     = 1e-5
	bnEps     = 1e-5
)

// Dims are SongEval's five scores, in its order.
var Dims = [5]string{"Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"}

// Scores are a song's five scores, each in (1, 5).
type Scores struct {
	Coherence, Musicality, Memorability, Clarity, Naturalness float64
}

func (s *Scores) dim(i int) *float64 {
	return [5]*float64{&s.Coherence, &s.Musicality, &s.Memorability, &s.Clarity, &s.Naturalness}[i]
}

// conv is one 3x3 convolution with its BatchNorm folded in: w [out][in][3][3]
// scaled per output channel, b the conv's bias through the norm.
type conv struct {
	w, b []float32
	in   int
}

type layerW struct {
	ffn1LNG, ffn1LNB, ffn2LNG, ffn2LNB []float32
	ffn1W1, ffn1B1, ffn1W2, ffn1B2     []float32
	ffn2W1, ffn2B1, ffn2W2, ffn2B2     []float32
	attnLNG, attnLNB                   []float32
	q, qB, k, kB, v, vB, o, oB         []float32
	convLNG, convLNB                   []float32
	pw1, pw2                           []float32 // [2048][1024], [1024][1024]
	dw                                 []float32 // [1024][31]
	bnScale, bnShift                   []float32
	finalLNG, finalLNB                 []float32
}

type weights struct {
	melMean, melStd float64
	// block 0: conv1, conv3 (from 1 channel), conv2; block 1: conv1, conv2, conv3.
	b0c1, b0c2, b0c3, b1c1, b1c2, b1c3 conv
	linW, linB                         []float32 // [1024][16384]
	invFreq                            []float32
	layer                              [Layers]layerW
	// SongEval's head.
	ffdW1, ffdB1, ffdW2, ffdB2 []float32
	inW, inB, outW, outB       [4][]float32 // in_proj [3072][1024]
	fcW, fcB                   []float32    // [5][2048]
}

type loader struct {
	set *safetensors.Set
	err error
}

func (l *loader) get(name string, n int) []float32 {
	if l.err != nil {
		return nil
	}
	t, err := l.set.Get(name)
	if err != nil {
		l.err = err
		return nil
	}
	v, err := t.F32(nil)
	if err != nil {
		l.err = err
		return nil
	}
	if n > 0 && len(v) != n {
		l.err = fmt.Errorf("songeval: %s has %d values, want %d", name, len(v), n)
	}
	return v
}

// bn folds a conv followed by BatchNorm (eval): y = (Wx + b − mean)·s + beta,
// s = gamma/sqrt(var + eps).
func (l *loader) bn(prefix, c, n string, in int) conv {
	w := l.get(prefix+c+".weight", convDim*in*9)
	b := l.get(prefix+c+".bias", convDim)
	g := l.get(prefix+n+".weight", convDim)
	beta := l.get(prefix+n+".bias", convDim)
	mean := l.get(prefix+n+".running_mean", convDim)
	vr := l.get(prefix+n+".running_var", convDim)
	if l.err != nil {
		return conv{}
	}
	out := conv{w: make([]float32, len(w)), b: make([]float32, convDim), in: in}
	for o := 0; o < convDim; o++ {
		s := float64(g[o]) / math.Sqrt(float64(vr[o])+bnEps)
		for i := 0; i < in*9; i++ {
			out.w[o*in*9+i] = float32(float64(w[o*in*9+i]) * s)
		}
		out.b[o] = float32((float64(b[o])-float64(mean[o]))*s + float64(beta[o]))
	}
	return out
}

func loadWeights(muqDir, headPath string) (*weights, error) {
	raw, err := os.ReadFile(filepath.Join(muqDir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Stat struct {
			Mean float64 `json:"melspec_2048_mean"`
			Std  float64 `json:"melspec_2048_std"`
		} `json:"stat"`
		Hop  int `json:"hop_length"`
		Mels int `json:"n_mels"`
		Dim  int `json:"encoder_dim"`
		W2V2 struct {
			Heads  int    `json:"num_attention_heads"`
			FFN    int    `json:"intermediate_size"`
			Kernel int    `json:"conv_depthwise_kernel_size"`
			Act    string `json:"hidden_act"`
			Pos    string `json:"position_embeddings_type"`
		} `json:"w2v2_config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("songeval: %s: %w", muqDir, err)
	}
	if cfg.Hop != hop || cfg.Mels != nMels || cfg.Dim != hidden || cfg.W2V2.Heads != heads || cfg.W2V2.FFN != ffn ||
		cfg.W2V2.Kernel != dwKernel || cfg.W2V2.Act != "swish" || cfg.W2V2.Pos != "rotary" {
		return nil, fmt.Errorf("songeval: MuQ config %+v is not the one this port is built for", cfg)
	}
	w := &weights{melMean: cfg.Stat.Mean, melStd: cfg.Stat.Std}

	set, err := safetensors.OpenSet(muqDir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	l := &loader{set: set}
	p := "model.conv.conv."
	w.b0c1 = l.bn(p+"0.", "conv1", "bn1", 1)
	w.b0c2 = l.bn(p+"0.", "conv2", "bn2", convDim)
	w.b0c3 = l.bn(p+"0.", "conv3", "bn3", 1)
	w.b1c1 = l.bn(p+"1.", "conv1", "bn1", convDim)
	w.b1c2 = l.bn(p+"1.", "conv2", "bn2", convDim)
	w.b1c3 = l.bn(p+"1.", "conv3", "bn3", convDim)
	w.linW = l.get("model.conv.linear.weight", hidden*convDim*nMels/4)
	w.linB = l.get("model.conv.linear.bias", hidden)
	w.invFreq = l.get("model.conformer.embed_positions.inv_freq", headDim/2)
	for i := range w.layer {
		lp := fmt.Sprintf("model.conformer.layers.%d.", i)
		L := &w.layer[i]
		L.ffn1LNG, L.ffn1LNB = l.get(lp+"ffn1_layer_norm.weight", hidden), l.get(lp+"ffn1_layer_norm.bias", hidden)
		L.ffn2LNG, L.ffn2LNB = l.get(lp+"ffn2_layer_norm.weight", hidden), l.get(lp+"ffn2_layer_norm.bias", hidden)
		L.ffn1W1, L.ffn1B1 = l.get(lp+"ffn1.intermediate_dense.weight", ffn*hidden), l.get(lp+"ffn1.intermediate_dense.bias", ffn)
		L.ffn1W2, L.ffn1B2 = l.get(lp+"ffn1.output_dense.weight", hidden*ffn), l.get(lp+"ffn1.output_dense.bias", hidden)
		L.ffn2W1, L.ffn2B1 = l.get(lp+"ffn2.intermediate_dense.weight", ffn*hidden), l.get(lp+"ffn2.intermediate_dense.bias", ffn)
		L.ffn2W2, L.ffn2B2 = l.get(lp+"ffn2.output_dense.weight", hidden*ffn), l.get(lp+"ffn2.output_dense.bias", hidden)
		L.attnLNG, L.attnLNB = l.get(lp+"self_attn_layer_norm.weight", hidden), l.get(lp+"self_attn_layer_norm.bias", hidden)
		for _, x := range []struct {
			w, b *[]float32
			n    string
		}{{&L.q, &L.qB, "linear_q"}, {&L.k, &L.kB, "linear_k"}, {&L.v, &L.vB, "linear_v"}, {&L.o, &L.oB, "linear_out"}} {
			*x.w, *x.b = l.get(lp+"self_attn."+x.n+".weight", hidden*hidden), l.get(lp+"self_attn."+x.n+".bias", hidden)
		}
		cm := lp + "conv_module."
		L.convLNG, L.convLNB = l.get(cm+"layer_norm.weight", hidden), l.get(cm+"layer_norm.bias", hidden)
		L.pw1 = l.get(cm+"pointwise_conv1.weight", 2*hidden*hidden)
		L.pw2 = l.get(cm+"pointwise_conv2.weight", hidden*hidden)
		L.dw = l.get(cm+"depthwise_conv.weight", hidden*dwKernel)
		g, b := l.get(cm+"batch_norm.weight", hidden), l.get(cm+"batch_norm.bias", hidden)
		mean, vr := l.get(cm+"batch_norm.running_mean", hidden), l.get(cm+"batch_norm.running_var", hidden)
		if l.err == nil {
			L.bnScale, L.bnShift = make([]float32, hidden), make([]float32, hidden)
			for c := 0; c < hidden; c++ {
				s := float64(g[c]) / math.Sqrt(float64(vr[c])+bnEps)
				L.bnScale[c] = float32(s)
				L.bnShift[c] = float32(float64(b[c]) - float64(mean[c])*s)
			}
		}
		L.finalLNG, L.finalLNB = l.get(lp+"final_layer_norm.weight", hidden), l.get(lp+"final_layer_norm.bias", hidden)
	}
	if l.err != nil {
		return nil, l.err
	}

	hs, err := safetensors.Open(headPath)
	if err != nil {
		return nil, err
	}
	defer hs.Close()
	var herr error
	hget := func(name string, n int) []float32 {
		if herr != nil {
			return nil
		}
		t, err := hs.Get(name)
		if err != nil {
			herr = err
			return nil
		}
		v, err := t.F32(nil)
		if err == nil && len(v) != n {
			err = fmt.Errorf("songeval: %s has %d values, want %d", name, len(v), n)
		}
		herr = err
		return v
	}
	w.ffdW1, w.ffdB1 = hget("ffd.0.weight", ffn*hidden), hget("ffd.0.bias", ffn)
	w.ffdW2, w.ffdB2 = hget("ffd.2.weight", hidden*ffn), hget("ffd.2.bias", hidden)
	for i := 0; i < 4; i++ {
		ap := fmt.Sprintf("attn.%d.", i)
		w.inW[i], w.inB[i] = hget(ap+"in_proj_weight", 3*hidden*hidden), hget(ap+"in_proj_bias", 3*hidden)
		w.outW[i], w.outB[i] = hget(ap+"out_proj.weight", hidden*hidden), hget(ap+"out_proj.bias", hidden)
	}
	w.fcW, w.fcB = hget("fc.weight", 5*2*hidden), hget("fc.bias", 5)
	if herr != nil {
		return nil, herr
	}
	return w, nil
}

// parallel splits [0, n) over the cores.
func parallel(n int, fn func(lo, hi int)) {
	wk := min(runtime.GOMAXPROCS(0), n)
	if wk <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + wk - 1) / wk
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
