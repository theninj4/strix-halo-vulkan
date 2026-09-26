package dit

import (
	"fmt"

	"strix-halo-vulkan/safetensors"
)

// Tables computes every block's AdaLN table for a request's distinct
// timesteps tvals, in fp32 on the host from the checkpoint's bf16
// projections: 13 B of the transformer's 33 B parameters, applied once per
// request and never staged on the device (VIDEO.md decision 3). Each block's
// projection is read, applied and dropped in turn, so the host holds one
// block's (1 GB as fp32) at a time.
func Tables(dir string, tvals []float32) ([]*Table, error) {
	host, err := Load(dir, 0)
	if err != nil {
		return nil, err
	}
	temb, err := host.TimeEmbed(tvals)
	if err != nil {
		return nil, err
	}
	act := siluMat(temb)
	set, err := safetensors.OpenSet(dir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	c := host.Cfg
	tabs := make([]*Table, c.Layers)
	for i := range tabs {
		l := &loader{set: set}
		lin := l.linear(fmt.Sprintf("transformer_blocks.%d.adaln_proj.linear", i), 6*Modalities*c.Hidden, c.TimeDim, true)
		if l.err != nil {
			return nil, l.err
		}
		out, err := lin.Apply(act)
		if err != nil {
			return nil, err
		}
		tabs[i] = &Table{Hidden: c.Hidden, Rows: temb.Rows * Modalities, Data: out.Data}
	}
	return tabs, nil
}
