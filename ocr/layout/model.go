// Package layout is PP-DocLayoutV3 (OCR.md O7): a page in, its regions out,
// each with one of 25 labels, a box, a score and a place in reading order.
//
// It is RT-DETR with two heads added. An HGNetV2-L backbone (stem, then four
// stages at strides 4, 8, 16 and 32), a hybrid encoder (one transformer
// layer over the stride-32 map, then a top-down FPN and a bottom-up PAN of
// CSP-RepVGG blocks), and a 6-layer decoder of 300 queries with deformable
// cross-attention over three levels. What is PP-DocLayoutV3's own:
//
//  1. **A mask head.** The encoder also builds 32 prototype maps at stride 4
//     (mask_feat, 200x200), and every query's mask is its 32-wide embedding
//     times them. The decoder's *initial boxes* come from these masks: the
//     bounding box of each selected query's mask > 0, not the encoder's box
//     head (mask_enhanced). A discrete step inside the model.
//  2. **A reading-order head**: the last layer's queries through a linear and
//     a global pointer give a [300, 300] matrix of "i before j" logits, and
//     the order is the ranking of their votes.
//  3. **The decoder's class and box heads are the encoder's** (tied, so the
//     checkpoint stores them once).
//
// This file loads the checkpoint (models/PP-DocLayoutV3, 858 fp32 tensors,
// the original checkpoint's names) with every batch norm folded into the
// conv before it; ops.go is the CPU arithmetic; forward.go the graph. The
// CPU path is the oracle the device port is debugged against, gated stage by
// stage against reference/dump_doclayout.py (HF's pp_doclayout_v3).
package layout

import (
	"fmt"
	"math"
	"path/filepath"

	"strix-halo-vulkan/safetensors"
)

const (
	dModel     = 256
	numQueries = 300
	numLabels  = 25
	numLevels  = 3
	numHeads   = 8
	numPoints  = 4
	numProto   = 32
	bnEps      = 1e-5
	lnEps      = 1e-5
	// InputSize is the square the page is resized to.
	InputSize = 800
)

// Labels are PaddleX's names (inference.yml's label_list), which keep apart
// what HF's config folds together: display_formula and inline_formula (both
// "formula" there), footer_image, header_image and vertical_text. The page
// pipeline's prompts and merges read these.
var Labels = [numLabels]string{
	"abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title",
	"figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image",
	"image", "inline_formula", "number", "paragraph_title", "reference", "reference_content", "seal",
	"table", "text", "vertical_text", "vision_footnote",
}

// basicLayer is one HGNetV2 block: a chain of convs whose outputs are all
// concatenated with the input, squeezed and excited back.
type basicLayer struct {
	layers   [][]*Conv // one conv, or a light block's 1x1 then depthwise
	squeeze  *Conv
	excite   *Conv
	residual bool
}

type stage struct {
	down   *Conv // nil for stage 1
	blocks []basicLayer
}

// repVGG is conv3x3 + conv1x1, summed, then SiLU.
type repVGG struct{ a, b *Conv }

// cspRep is PPDocLayoutV3CSPRepLayer.
type cspRep struct {
	conv1, conv2 *Conv
	bottlenecks  [3]repVGG
}

type attnLayer struct {
	q, k, v, o Linear
	ln1        LayerNorm
	fc1, fc2   Linear
	ln2        LayerNorm
}

type decLayer struct {
	self             attnLayer // q, k, v, o and ln1 (self_attn_layer_norm)
	offsets, weights Linear
	value, output    Linear
	lnCross          LayerNorm
	fc1, fc2         Linear
	lnFinal          LayerNorm
}

type mlp []Linear // ReLU between layers

// Model is PP-DocLayoutV3, fp32, on the host.
type Model struct {
	stem    [5]*Conv // stem1, 2a, 2b, 3, 4
	stages  [4]stage
	encProj [3]*Conv // stages 2-4 to 256

	aifi       attnLayer
	lateral    [2]*Conv
	fpn        [2]cspRep
	downsample [2]*Conv
	pan        [2]cspRep

	scaleHeads [3][]*Conv // strides 8, 16, 32: 1, 1 and 2 convs, each followed by a 2x upsample past the first head
	maskOut    *Conv      // mask_feature_head.output_conv
	maskLat    *Conv      // encoder_mask_lateral (the stride-4 backbone map)
	maskBase   *Conv      // encoder_mask_output.base_conv
	maskProto  *Conv      // encoder_mask_output.conv, 64 -> 32 with bias

	decProj [3]*Conv

	encOutput Linear
	encOutLN  LayerNorm
	scoreHead Linear // enc_score_head = decoder.class_embed
	bboxHead  mlp    // enc_bbox_head = decoder.bbox_embed
	queryPos  mlp
	layers    [6]decLayer
	orderHead [6]Linear
	pointer   Linear // global pointer's dense, 256 -> 2x64
	decNorm   LayerNorm
	maskQuery mlp
}

// Load reads dir/model.safetensors.
func Load(dir string) (*Model, error) {
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lerr error
	get := func(name string) ([]float32, []int) {
		if lerr != nil {
			return nil, nil
		}
		t, err := f.Get(name)
		if err != nil {
			lerr = err
			return nil, nil
		}
		v, err := t.F32(nil)
		if err != nil {
			lerr = err
		}
		return v, t.Shape
	}
	// conv reads a conv weight and folds the batch norm at bn (a prefix, or
	// "" for none) into it.
	conv := func(w, bias, bn string, stride, groups int, act Act) *Conv {
		wv, shape := get(w)
		if lerr != nil {
			return nil
		}
		c := &Conv{Out: shape[0], In: shape[1] * groups, K: shape[2], Stride: stride, Groups: groups, W: wv, Act: act}
		c.B = make([]float32, c.Out)
		if bias != "" {
			b, _ := get(bias)
			copy(c.B, b)
		}
		if bn != "" {
			g, _ := get(bn + ".weight")
			be, _ := get(bn + ".bias")
			mu, _ := get(bn + ".running_mean")
			va, _ := get(bn + ".running_var")
			if lerr != nil {
				return nil
			}
			per := len(wv) / c.Out
			for o := 0; o < c.Out; o++ {
				s := float64(g[o]) / math.Sqrt(float64(va[o])+bnEps)
				for i := o * per; i < (o+1)*per; i++ {
					wv[i] = float32(float64(wv[i]) * s)
				}
				c.B[o] = float32((float64(c.B[o])-float64(mu[o]))*s + float64(be[o]))
			}
		}
		return c
	}
	// hg is an HGNetV2ConvLayer: padding (k-1)/2.
	hg := func(p string, stride, groups int, act Act) *Conv {
		c := conv(p+".convolution.weight", "", p+".normalization", stride, groups, act)
		if c != nil {
			c.Pad = (c.K - 1) / 2
		}
		return c
	}
	// cn is a ConvNormLayer (conv/norm names): padding (k-1)/2.
	cn := func(p string, stride int, act Act) *Conv {
		c := conv(p+".conv.weight", "", p+".norm", stride, 1, act)
		if c != nil {
			c.Pad = (c.K - 1) / 2
		}
		return c
	}
	// cl is a PPDocLayoutV3ConvLayer: padding k/2.
	cl := func(p string, act Act) *Conv {
		c := conv(p+".convolution.weight", "", p+".normalization", 1, 1, act)
		if c != nil {
			c.Pad = c.K / 2
		}
		return c
	}
	lin := func(p string) Linear {
		w, shape := get(p + ".weight")
		b, _ := get(p + ".bias")
		if lerr != nil {
			return Linear{}
		}
		return Linear{Out: shape[0], In: shape[1], W: w, B: b}
	}
	ln := func(p string) LayerNorm {
		w, _ := get(p + ".weight")
		b, _ := get(p + ".bias")
		return LayerNorm{W: w, B: b, Eps: lnEps}
	}
	mlpOf := func(p string, n int) mlp {
		var m mlp
		for i := 0; i < n; i++ {
			m = append(m, lin(fmt.Sprintf("%s.layers.%d", p, i)))
		}
		return m
	}

	m := &Model{}
	const bb = "model.backbone.model."
	for i, s := range []struct {
		name   string
		stride int
	}{{"stem1", 2}, {"stem2a", 1}, {"stem2b", 1}, {"stem3", 2}, {"stem4", 1}} {
		m.stem[i] = hg(bb+"embedder."+s.name, s.stride, 1, ActReLU)
	}
	blocks := [4]int{1, 1, 3, 1}
	light := [4]bool{false, false, true, true}
	for s := 0; s < 4; s++ {
		p := fmt.Sprintf(bb+"encoder.stages.%d.", s)
		st := &m.stages[s]
		if s > 0 {
			// Depthwise 3x3 stride 2 over the stage's input, batch norm, no
			// activation.
			st.down = hg(p+"downsample", 2, [4]int{48, 128, 512, 1024}[s], ActNone)
		}
		for b := 0; b < blocks[s]; b++ {
			bp := fmt.Sprintf("%sblocks.%d.", p, b)
			bl := basicLayer{residual: b != 0}
			for l := 0; l < 6; l++ {
				lp := fmt.Sprintf("%slayers.%d", bp, l)
				if light[s] {
					c1 := hg(lp+".conv1", 1, 1, ActNone)
					var groups int
					if c1 != nil {
						groups = c1.Out
					}
					bl.layers = append(bl.layers, []*Conv{c1, hg(lp+".conv2", 1, groups, ActReLU)})
				} else {
					bl.layers = append(bl.layers, []*Conv{hg(lp, 1, 1, ActReLU)})
				}
			}
			bl.squeeze = hg(bp+"aggregation.0", 1, 1, ActReLU)
			bl.excite = hg(bp+"aggregation.1", 1, 1, ActReLU)
			st.blocks = append(st.blocks, bl)
		}
	}
	for i := range m.encProj {
		p := fmt.Sprintf("model.encoder_input_proj.%d", i)
		m.encProj[i] = conv(p+".0.weight", "", p+".1", 1, 1, ActNone)
	}

	const enc = "model.encoder."
	attn := func(p, o string) attnLayer {
		return attnLayer{
			q: lin(p + "self_attn.q_proj"), k: lin(p + "self_attn.k_proj"), v: lin(p + "self_attn.v_proj"),
			o: lin(p + "self_attn." + o), ln1: ln(p + "self_attn_layer_norm"),
			fc1: lin(p + "fc1"), fc2: lin(p + "fc2"), ln2: ln(p + "final_layer_norm"),
		}
	}
	m.aifi = attn(enc+"encoder.0.layers.0.", "out_proj")
	csp := func(p string) cspRep {
		c := cspRep{conv1: cn(p+".conv1", 1, ActSiLU), conv2: cn(p+".conv2", 1, ActSiLU)}
		for i := range c.bottlenecks {
			bp := fmt.Sprintf("%s.bottlenecks.%d", p, i)
			c.bottlenecks[i] = repVGG{a: cn(bp+".conv1", 1, ActNone), b: cn(bp+".conv2", 1, ActNone)}
		}
		return c
	}
	for i := 0; i < 2; i++ {
		m.lateral[i] = cn(fmt.Sprintf(enc+"lateral_convs.%d", i), 1, ActSiLU)
		m.fpn[i] = csp(fmt.Sprintf(enc+"fpn_blocks.%d", i))
		m.downsample[i] = cn(fmt.Sprintf(enc+"downsample_convs.%d", i), 2, ActSiLU)
		m.pan[i] = csp(fmt.Sprintf(enc+"pan_blocks.%d", i))
	}
	for h, idx := range [3][]int{{0}, {0}, {0, 2}} {
		for _, j := range idx {
			m.scaleHeads[h] = append(m.scaleHeads[h], cl(fmt.Sprintf(enc+"mask_feature_head.scale_heads.%d.layers.%d", h, j), ActSiLU))
		}
	}
	m.maskOut = cl(enc+"mask_feature_head.output_conv", ActSiLU)
	m.maskLat = cl(enc+"encoder_mask_lateral", ActSiLU)
	m.maskBase = cl(enc+"encoder_mask_output.base_conv", ActSiLU)
	m.maskProto = conv(enc+"encoder_mask_output.conv.weight", enc+"encoder_mask_output.conv.bias", "", 1, 1, ActNone)

	for i := range m.decProj {
		p := fmt.Sprintf("model.decoder_input_proj.%d", i)
		m.decProj[i] = conv(p+".0.weight", "", p+".1", 1, 1, ActNone)
	}
	m.encOutput = lin("model.enc_output.0")
	m.encOutLN = ln("model.enc_output.1")
	m.scoreHead = lin("model.enc_score_head")
	m.bboxHead = mlpOf("model.enc_bbox_head", 3)
	m.queryPos = mlpOf("model.decoder.query_pos_head", 2)
	for i := range m.layers {
		p := fmt.Sprintf("model.decoder.layers.%d.", i)
		a := attn(p, "out_proj")
		m.layers[i] = decLayer{
			self:    a,
			offsets: lin(p + "encoder_attn.sampling_offsets"), weights: lin(p + "encoder_attn.attention_weights"),
			value: lin(p + "encoder_attn.value_proj"), output: lin(p + "encoder_attn.output_proj"),
			lnCross: ln(p + "encoder_attn_layer_norm"),
			fc1:     a.fc1, fc2: a.fc2, lnFinal: a.ln2,
		}
		m.orderHead[i] = lin(fmt.Sprintf("model.decoder_order_head.%d", i))
	}
	m.pointer = lin("model.decoder_global_pointer.dense")
	m.decNorm = ln("model.decoder_norm")
	m.maskQuery = mlpOf("model.mask_query_head", 3)
	if lerr != nil {
		return nil, fmt.Errorf("layout: %w", lerr)
	}
	return m, nil
}

func (m mlp) apply(x *Rows) *Rows {
	for i := range m {
		x = m[i].Apply(x)
		if i < len(m)-1 {
			actRows(x, ActReLU)
		}
	}
	return x
}
