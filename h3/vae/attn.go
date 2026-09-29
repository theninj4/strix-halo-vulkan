package vae

import (
	"fmt"

	"strix-halo-vulkan/h3/dit"
	"strix-halo-vulkan/shaders"
)

// AttnVariant is one build of the decoder's attention at head 64: QT query
// tiles a wave, KTIL key tiles a block, T the transposed kernel
// (h3_attn_t.comp, VIDEO.md M11c/M11e) rather than dit_attention_wmma.comp.
type AttnVariant struct {
	QT, KTIL int
	T        bool
}

func (v AttnVariant) String() string {
	k := "plain"
	if v.T {
		k = "transposed"
	}
	return fmt.Sprintf("%s QT%d KTIL%d", k, v.QT, v.KTIL)
}

var (
	attnPlain = AttnVariant{QT: 1, KTIL: 4}
	// attnT is the transposed build a device with h3_attn_t.comp's element
	// order runs (TestGPUAttentionScreen, M11e).
	attnT = AttnVariant{QT: 2, KTIL: 4, T: true}
)

var attnBuilds = map[AttnVariant][]byte{
	attnPlain:                 shaders.H3VAEAttnHD64,
	{QT: 1, KTIL: 4, T: true}: shaders.H3VAEAttnTHD64QT1KT4,
	{QT: 2, KTIL: 4, T: true}: shaders.H3VAEAttnTHD64QT2KT4,
	{QT: 4, KTIL: 4, T: true}: shaders.H3VAEAttnTHD64QT4KT4,
	{QT: 1, KTIL: 8, T: true}: shaders.H3VAEAttnTHD64QT1KT8,
	{QT: 2, KTIL: 8, T: true}: shaders.H3VAEAttnTHD64QT2KT8,
}

// chooseAttention probes the device's element order at the first run, as
// h3/dit does at its first Begin, and takes the transposed build if it is
// the one h3_attn_t.comp is written against.
func (g *GPU) chooseAttention() error {
	if g.attnProbed || g.attnFixed {
		return nil
	}
	ok, err := dit.TransposedAttentionOK(g.dev)
	if err != nil {
		return err
	}
	if ok {
		g.attn = attnT
	}
	g.attnProbed = true
	return nil
}

// SetAttention pins the attention build (a screen, or a control arm).
func (g *GPU) SetAttention(v AttnVariant) error {
	if _, ok := g.attnPipes[v]; !ok {
		return fmt.Errorf("vae: no attention build %v", v)
	}
	g.attn, g.attnFixed = v, true
	return nil
}

// AttnVariants lists the builds a screen chooses between.
func AttnVariants() []AttnVariant {
	out := make([]AttnVariant, 0, len(attnBuilds))
	for v := range attnBuilds {
		out = append(out, v)
	}
	return out
}
