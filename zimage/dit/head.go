package dit

import (
	"fmt"
	"math"

	"strix-halo-vulkan/safetensors"
)

// Head is everything the transformer does around its 34 blocks: the timestep
// embedding every modulated block is conditioned on, the patch embedder that
// turns a latent into image tokens, the caption embedder that turns the text
// encoder's hidden state into caption tokens, the two learned pad tokens, and
// the final layer that turns the residual stream back into a latent.
//
// It is small -- 130 MB against the blocks' 24.6 GB -- and it is where the
// model's index arithmetic lives, which is the opposite balance from
// everything before it: nothing here is worth a shader and everything here is
// worth a test. The four things that produce a plausible tensor when they are
// wrong:
//
//   - The caption is padded to a multiple of SeqMultiOf and the padded rows
//     are replaced by a learned token *after* the embedder, not before.
//   - The caption's position ids start at 1, and the image's axis-0 id is the
//     *padded* caption length plus one, so the same prompt at 40 and at 64
//     tokens puts the image at different rotary positions.
//   - Patchify interleaves (ph, pw, c) into one feature vector; the obvious
//     (c, ph, pw) produces a latent that decodes to noise.
//   - The final layer's norm is a LayerNorm -- the mean is subtracted -- where
//     every other norm in this model is an RMS norm.
//
// reference/dump_zimage.py is the oracle for all of it.
type Head struct {
	Dim, InChan     int
	Patch, FPatch   int
	CapFeat, AdaDim int
	TScale          float64

	// t_embedder: Linear(256, 1024), SiLU, Linear(1024, 256), over the
	// sinusoidal embedding of t*TScale.
	TMLP0, TMLP2 *Linear
	FreqDim      int

	// all_x_embedder.<patch>-<f_patch>: Linear(pF*pH*pW*C, dim).
	XEmbed *Linear

	// cap_embedder: RMSNorm(cap_feat_dim), Linear(cap_feat_dim, dim).
	CapNorm *RMSNorm
	CapProj *Linear

	// The learned tokens that overwrite a padded row of either stream.
	XPad, CapPad []float32

	// all_final_layer.<patch>-<f_patch>: LayerNorm without affine, scaled by
	// 1 + Linear(SiLU(adaln)), then Linear(dim, pF*pH*pW*C).
	FinalAda *Linear
	FinalLin *Linear
	FinalEps float64
}

// SeqMultiOf is the length both token streams are padded up to, diffusers'
// SEQ_MULTI_OF. A 1024x1024 image is 4096 tokens and needs no padding; a
// caption almost always does.
const SeqMultiOf = 32

// finalNormEps is the epsilon diffusers hardcodes on the final layer's
// LayerNorm, which is not the config's norm_eps: 1e-6 against 1e-5.
const finalNormEps = 1e-6

// freqEmbedDim is TimestepEmbedder's frequency_embedding_size.
const freqEmbedDim = 256

// maxPeriod is the sinusoidal embedding's longest period.
const maxPeriod = 10000

// LoadHead reads the non-block tensors out of a transformer checkpoint.
func LoadHead(set *safetensors.Set, cfg *Config) (*Head, error) {
	if len(cfg.PatchSize) == 0 {
		return nil, fmt.Errorf("dit: config.json has no all_patch_size")
	}
	patch, fPatch := cfg.PatchSize[0], 1
	key := fmt.Sprintf("%d-%d", patch, fPatch)

	l := &loader{set: set}
	h := &Head{
		Dim: cfg.Dim, InChan: cfg.InChan, Patch: patch, FPatch: fPatch,
		CapFeat: cfg.CapFeat, TScale: cfg.TScale, FreqDim: freqEmbedDim,
		TMLP0:    l.linear("t_embedder.mlp.0"),
		TMLP2:    l.linear("t_embedder.mlp.2"),
		XEmbed:   l.linear("all_x_embedder." + key),
		CapNorm:  l.rms("cap_embedder.0", cfg.NormEps),
		CapProj:  l.linear("cap_embedder.1"),
		XPad:     l.f32("x_pad_token"),
		CapPad:   l.f32("cap_pad_token"),
		FinalAda: l.linear("all_final_layer." + key + ".adaLN_modulation.1"),
		FinalLin: l.linear("all_final_layer." + key + ".linear"),
		FinalEps: finalNormEps,
	}
	if l.err != nil {
		return nil, l.err
	}
	h.AdaDim = h.TMLP2.Out
	if want := fPatch * patch * patch * cfg.InChan; h.XEmbed.In != want {
		return nil, fmt.Errorf("dit: x_embedder takes %d features, want pF*pH*pW*C = %d", h.XEmbed.In, want)
	}
	if h.FinalLin.Out != h.XEmbed.In {
		return nil, fmt.Errorf("dit: final layer produces %d, want the patch's %d", h.FinalLin.Out, h.XEmbed.In)
	}
	if len(h.XPad) != cfg.Dim || len(h.CapPad) != cfg.Dim {
		return nil, fmt.Errorf("dit: pad tokens are %d and %d wide, want dim %d", len(h.XPad), len(h.CapPad), cfg.Dim)
	}
	return h, nil
}

// PatchDim is how many latent values one image token carries.
func (h *Head) PatchDim() int { return h.FPatch * h.Patch * h.Patch * h.InChan }

// Timestep is the adaLN conditioning vector for one denoising step. t is what
// the pipeline hands the transformer -- (1000 - timestep)/1000, i.e. 1-sigma
// -- and TScale takes it back to the model's own units.
func (h *Head) Timestep(t float64) ([]float32, error) {
	freq := SinusoidalEmbedding(t*h.TScale, h.FreqDim)
	in := &Mat{Rows: 1, Cols: len(freq), Data: freq}
	mid, err := h.TMLP0.Apply(in)
	if err != nil {
		return nil, err
	}
	for i, v := range mid.Data {
		mid.Data[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
	out, err := h.TMLP2.Apply(mid)
	if err != nil {
		return nil, err
	}
	return out.Data, nil
}

// SinusoidalEmbedding is diffusers' TimestepEmbedder.timestep_embedding: the
// cosines of dim/2 geometrically spaced frequencies followed by their sines.
//
// The halves are cos-then-sin, which is the opposite of the sin-then-cos most
// diffusion code uses, and swapping them is invisible in every norm.
func SinusoidalEmbedding(t float64, dim int) []float32 {
	half := dim / 2
	out := make([]float32, dim)
	for j := 0; j < half; j++ {
		freq := math.Exp(-math.Log(maxPeriod) * float64(j) / float64(half))
		arg := t * freq
		out[j] = float32(math.Cos(arg))
		out[half+j] = float32(math.Sin(arg))
	}
	if dim%2 != 0 {
		out[dim-1] = 0
	}
	return out
}

// Patchify turns a latent [C, H, W] into image tokens [H/p * W/p, p*p*C].
//
// The feature order inside a token is (ph, pw, c) with the channel varying
// fastest: diffusers reshapes (C, F, H, W) to (C, F_t, pF, H_t, pH, W_t, pW)
// and permutes to (F_t, H_t, W_t, pF, pH, pW, C). Putting the channel first
// instead type-checks, round-trips through Unpatchify, and decodes to noise.
func (h *Head) Patchify(latent []float32, height, width int) (*Mat, error) {
	p, c := h.Patch, h.InChan
	if height%p != 0 || width%p != 0 {
		return nil, fmt.Errorf("dit: latent %dx%d is not a multiple of the patch size %d", height, width, p)
	}
	if len(latent) != c*height*width {
		return nil, fmt.Errorf("dit: latent is %d values, want C*H*W = %d", len(latent), c*height*width)
	}
	ht, wt := height/p, width/p
	out := NewMat(ht*wt, h.PatchDim())
	for th := 0; th < ht; th++ {
		for tw := 0; tw < wt; tw++ {
			row := out.Row(th*wt + tw)
			for ph := 0; ph < p; ph++ {
				for pw := 0; pw < p; pw++ {
					src := (th*p+ph)*width + tw*p + pw
					dst := (ph*p + pw) * c
					for ch := 0; ch < c; ch++ {
						row[dst+ch] = latent[ch*height*width+src]
					}
				}
			}
		}
	}
	return out, nil
}

// Unpatchify is Patchify's inverse over the final layer's output: the first
// ht*wt rows of x become a latent [C, H, W]. Rows past them are the padding
// and the caption, which the model produces and the pipeline discards.
func (h *Head) Unpatchify(x *Mat, height, width int) ([]float32, error) {
	p, c := h.Patch, h.InChan
	if height%p != 0 || width%p != 0 {
		return nil, fmt.Errorf("dit: latent %dx%d is not a multiple of the patch size %d", height, width, p)
	}
	if x.Cols != h.PatchDim() {
		return nil, fmt.Errorf("dit: final output is %s, want %d columns", x, h.PatchDim())
	}
	ht, wt := height/p, width/p
	if x.Rows < ht*wt {
		return nil, fmt.Errorf("dit: final output has %d rows, want at least %d image tokens", x.Rows, ht*wt)
	}
	out := make([]float32, c*height*width)
	for th := 0; th < ht; th++ {
		for tw := 0; tw < wt; tw++ {
			row := x.Row(th*wt + tw)
			for ph := 0; ph < p; ph++ {
				for pw := 0; pw < p; pw++ {
					dst := (th*p+ph)*width + tw*p + pw
					src := (ph*p + pw) * c
					for ch := 0; ch < c; ch++ {
						out[ch*height*width+dst] = row[src+ch]
					}
				}
			}
		}
	}
	return out, nil
}

// PadTo is the length a stream of n tokens is padded to.
func PadTo(n int) int { return (n + SeqMultiOf - 1) / SeqMultiOf * SeqMultiOf }

// EmbedImage runs the patch embedder and pads the stream with x_pad_token.
func (h *Head) EmbedImage(patches *Mat) (*Mat, error) {
	x, err := h.XEmbed.Apply(patches)
	if err != nil {
		return nil, err
	}
	return padStream(x, h.XPad), nil
}

// EmbedCaption runs the caption embedder over the text encoder's hidden state
// and pads the stream with cap_pad_token.
//
// The padding is applied to the *embedded* rows, so what the padded positions
// carry is the learned token and not the embedding of anything; diffusers
// pads the input by repeating its last row first, which nothing then reads.
func (h *Head) EmbedCaption(cap *Mat) (*Mat, error) {
	if cap.Cols != h.CapFeat {
		return nil, fmt.Errorf("dit: caption features are %s, want %d columns", cap, h.CapFeat)
	}
	normed := cap.Clone()
	if _, err := h.CapNorm.ApplyInPlace(normed); err != nil {
		return nil, err
	}
	out, err := h.CapProj.Apply(normed)
	if err != nil {
		return nil, err
	}
	return padStream(out, h.CapPad), nil
}

// padStream extends x to a multiple of SeqMultiOf with copies of tok.
func padStream(x *Mat, tok []float32) *Mat {
	total := PadTo(x.Rows)
	if total == x.Rows {
		return x
	}
	out := NewMat(total, x.Cols)
	copy(out.Data, x.Data)
	for r := x.Rows; r < total; r++ {
		copy(out.Row(r), tok)
	}
	return out
}

// PositionIDs is the 3-D rotary grid for the unified sequence, in its own
// order: the image tokens first, then the caption.
//
// capTokens is the caption's length *before* padding; the padded length is
// what the image's axis-0 position is measured from, and the caption's own
// padded rows carry the ids that follow it rather than a repeat. Image
// padding, if the grid is not a multiple of SeqMultiOf, gets the origin.
func (h *Head) PositionIDs(capTokens, height, width int) ([][3]int32, error) {
	p := h.Patch
	if height%p != 0 || width%p != 0 {
		return nil, fmt.Errorf("dit: latent %dx%d is not a multiple of the patch size %d", height, width, p)
	}
	ht, wt := height/p, width/p
	capTotal := PadTo(capTokens)
	imgTotal := PadTo(ht * wt)

	ids := make([][3]int32, 0, imgTotal+capTotal)
	// The image tiles axes 1 and 2 at a single axis-0 position, one past the
	// caption it is conditioned on.
	base := int32(capTotal + 1)
	for y := 0; y < ht; y++ {
		for x := 0; x < wt; x++ {
			ids = append(ids, [3]int32{base, int32(y), int32(x)})
		}
	}
	for i := ht * wt; i < imgTotal; i++ {
		ids = append(ids, [3]int32{0, 0, 0})
	}
	// The caption advances along axis 0 from 1. Its padded rows continue the
	// count -- diffusers builds the ids from the padded length and appends
	// pad ids past the end of the stream, which _prepare_sequence truncates.
	for i := 0; i < capTotal; i++ {
		ids = append(ids, [3]int32{int32(i + 1), 0, 0})
	}
	return ids, nil
}

// Final is the transformer's tail: a LayerNorm without affine, scaled by
// 1 + adaLN(SiLU(adaln)), then projected back to a patch.
func (h *Head) Final(x *Mat, adaln []float32) (*Mat, error) {
	if x.Cols != h.Dim {
		return nil, fmt.Errorf("dit: final layer takes %d features, got %s", h.Dim, x)
	}
	if len(adaln) != h.FinalAda.In {
		return nil, fmt.Errorf("dit: final adaLN takes %d, got %d", h.FinalAda.In, len(adaln))
	}
	act := make([]float32, len(adaln))
	for i, v := range adaln {
		act[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
	scaleMat, err := h.FinalAda.Apply(&Mat{Rows: 1, Cols: len(act), Data: act})
	if err != nil {
		return nil, err
	}
	scale := scaleMat.Data
	normed := NewMat(x.Rows, x.Cols)
	parallelFor(x.Rows, func(r int) {
		src, dst := x.Row(r), normed.Row(r)
		var mean float64
		for _, v := range src {
			mean += float64(v)
		}
		mean /= float64(len(src))
		var variance float64
		for _, v := range src {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(len(src))
		inv := 1 / math.Sqrt(variance+h.FinalEps)
		for i, v := range src {
			dst[i] = float32((float64(v)-mean)*inv) * (1 + scale[i])
		}
	})
	return h.FinalLin.Apply(normed)
}
