// Package vae is MiniMax-H3's video VAE decoder (VIDEO.md M5): a 36-layer
// ViT that turns 24-channel latents at 4x16x16 compression back into
// ImageNet-normalised RGB.
//
// This file is the host side, and none of it is heavy: the config, the
// latent and video tensors, the post_quant_conv (a 1x1 conv, pointwise), the
// tiling and the temporal clipping that decide which latent blocks the
// decoder sees, and the linear cross-fades that put its outputs back
// together. Every one of these is reproduced from diffusers'
// `AutoencoderKLMiniMaxH3` rather than re-derived, because the released
// frames are the blended-tile ones: tiling is on by default and changes the
// output. The decoder itself runs on the device (gpu.go).
package vae

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

// Config is vae/config.json, the decoder's half of it.
type Config struct {
	LatentChannels  int       `json:"latent_channels"`
	OutChannels     int       `json:"out_channels"`
	SpatialFactors  []int     `json:"spatial_downsample_factors"`
	TemporalFactors []int     `json:"temporal_downsample_factors"`
	Layers          int       `json:"decoder_num_layers"`
	Heads           int       `json:"decoder_num_attention_heads"`
	HeadDim         int       `json:"decoder_attention_head_dim"`
	Registers       int       `json:"decoder_num_register_tokens"`
	FFNMult         int       `json:"decoder_ffn_mult"`
	RopeTheta       float64   `json:"decoder_rope_theta"`
	RopeRatio       float64   `json:"decoder_rope_dim_ratio"`
	NormEps         float64   `json:"decoder_norm_eps"`
	ClipLength      int       `json:"clip_length"`
	TokenDrop       int       `json:"token_drop"`
	LatentsMean     []float32 `json:"latents_mean"`
	LatentsStd      []float32 `json:"latents_std"`
}

// LoadConfig reads dir/config.json (dir is the checkpoint's vae/).
func LoadConfig(dir string) (*Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	c := &Config{}
	if err := json.Unmarshal(raw, c); err != nil {
		return nil, fmt.Errorf("vae: %s: %w", dir, err)
	}
	if c.Heads*c.HeadDim == 0 || c.Layers == 0 || len(c.LatentsMean) != c.LatentChannels || len(c.LatentsStd) != c.LatentChannels {
		return nil, fmt.Errorf("vae: %s is not a MiniMax-H3 video VAE config", dir)
	}
	return c, nil
}

// Hidden is the decoder's width, FFN its SwiGLU's inner width.
func (c *Config) Hidden() int { return c.Heads * c.HeadDim }
func (c *Config) FFN() int    { return c.Hidden() * c.FFNMult }

// Spatial and Temporal are the compression ratios: the decoder's patch.
func (c *Config) Spatial() int  { return prod(c.SpatialFactors) }
func (c *Config) Temporal() int { return prod(c.TemporalFactors) }

// RopeWidth is how many of a head's channels are rotated: 48 of 64.
func (c *Config) RopeWidth() int { return int(float64(c.HeadDim) * c.RopeRatio) }

// PatchOut is proj_out's width: 3 × 4 × 16 × 16.
func (c *Config) PatchOut() int { return c.OutChannels * c.Temporal() * c.Spatial() * c.Spatial() }

func prod(xs []int) int {
	p := 1
	for _, x := range xs {
		p *= x
	}
	return p
}

// The temporal geometry diffusers derives from clip_length and token_drop
// (`AutoencoderKLMiniMaxH3.__init__`). The encoder saw clip_length pixel
// frames at a time and dropped the last token_drop latents of each, so the
// decoder reads clips of chunkTokens + tokenOverlap latents that overlap by
// tokenOverlap, and cross-fades frameOverlap pixel frames between them.
func (c *Config) prePadding() int   { return mod(-c.ClipLength, c.Temporal()) }                 // 3
func (c *Config) chunkTokens() int  { return (c.ClipLength + c.Temporal() - 1) / c.Temporal() } // 5
func (c *Config) tokenOverlap() int { return mod(-c.TokenDrop, c.chunkTokens()) }               // 2
func (c *Config) frameOverlap() int {
	return max(c.tokenOverlap()*c.Temporal()-c.prePadding(), 0) // 5
}

// ClipTokens is how many latent frames one decoder call reads: 7.
func (c *Config) ClipTokens() int { return c.chunkTokens() + c.tokenOverlap() }

func mod(a, m int) int { return ((a % m) + m) % m }

// Frames is how many pixel frames lf latent frames decode to: 17n + 5 for
// the 5n + 2 latents the pipeline makes.
func (c *Config) Frames(lf int) (int, error) {
	n, err := c.clips(lf)
	if err != nil {
		return 0, err
	}
	return n*(c.chunkTokens()*c.Temporal()-c.prePadding()) + c.frameOverlap(), nil
}

// clips is how many decoder clips lf latent frames take. diffusers repeats
// the last latent when lf + token_drop is not a whole number of chunks, and
// cuts the extra frames off again; the pipeline never makes such a count
// (5n + 2 always), so that path is refused rather than carried.
func (c *Config) clips(lf int) (int, error) {
	tokens := lf + c.TokenDrop
	if c.TokenDrop == 0 || tokens%c.chunkTokens() != 0 || tokens/c.chunkTokens() < 2 {
		return 0, fmt.Errorf("vae: %d latent frames is not %dn + %d", lf, c.chunkTokens(), c.chunkTokens()-c.TokenDrop)
	}
	return tokens/c.chunkTokens() - 1, nil
}

// Tensor is a [C, T, H, W] float32 volume: a latent or a video.
type Tensor struct {
	C, T, H, W int
	Data       []float32
}

func NewTensor(c, t, h, w int) *Tensor {
	return &Tensor{C: c, T: t, H: h, W: w, Data: make([]float32, c*t*h*w)}
}

func (x *Tensor) idx(c, t, h, w int) int { return ((c*x.T+t)*x.H+h)*x.W + w }

// At and Set index the volume.
func (x *Tensor) At(c, t, h, w int) float32     { return x.Data[x.idx(c, t, h, w)] }
func (x *Tensor) Set(c, t, h, w int, v float32) { x.Data[x.idx(c, t, h, w)] = v }

// Slice copies [t0, t0+nt) × [h0, h0+nh) × [w0, w0+nw) of every channel.
func (x *Tensor) Slice(t0, nt, h0, nh, w0, nw int) *Tensor {
	out := NewTensor(x.C, nt, nh, nw)
	for c := 0; c < x.C; c++ {
		for t := 0; t < nt; t++ {
			for h := 0; h < nh; h++ {
				src := x.idx(c, t0+t, h0+h, w0)
				copy(out.Data[out.idx(c, t, h, 0):out.idx(c, t, h, nw)], x.Data[src:src+nw])
			}
		}
	}
	return out
}

// Unpatchify turns the transformer's video rows — [lf·(lh/2)·(lw/2), 96],
// rows frame-major then row-major, columns (channel, 1, 2, 2) — into the
// latent volume, and Denormalize undoes the per-channel normalisation the
// pipeline works in: z·std + mean, which is what the VAE decodes.
func Unpatchify(rows []float32, channels, lf, lh, lw int) (*Tensor, error) {
	if len(rows) != channels*lf*lh*lw || lh%2 != 0 || lw%2 != 0 {
		return nil, fmt.Errorf("vae: %d values for a %dx%dx%dx%d latent", len(rows), channels, lf, lh, lw)
	}
	z := NewTensor(channels, lf, lh, lw)
	gh, gw := lh/2, lw/2
	i := 0
	for f := 0; f < lf; f++ {
		for y := 0; y < gh; y++ {
			for x := 0; x < gw; x++ {
				for c := 0; c < channels; c++ {
					for py := 0; py < 2; py++ {
						for px := 0; px < 2; px++ {
							z.Set(c, f, 2*y+py, 2*x+px, rows[i])
							i++
						}
					}
				}
			}
		}
	}
	return z, nil
}

func (c *Config) Denormalize(z *Tensor) {
	plane := z.T * z.H * z.W
	for ch := 0; ch < z.C; ch++ {
		s, m := c.LatentsStd[ch], c.LatentsMean[ch]
		for i, v := range z.Data[ch*plane : (ch+1)*plane] {
			z.Data[ch*plane+i] = float32(v*s) + m
		}
	}
}

// ImageNet's statistics: the decoder's output is (rgb − mean)/std over a
// [0, 1] base.
var (
	PixelMean = [3]float32{0.485, 0.456, 0.406}
	PixelStd  = [3]float32{0.229, 0.224, 0.225}
)

// ToRGB8 converts a decoded video to 8-bit RGB frames, [T][H·W·3]: the
// pipeline's (x·std + mean).clamp(0, 1), then ×255 rounded.
func ToRGB8(v *Tensor) [][]byte {
	out := make([][]byte, v.T)
	for t := range out {
		f := make([]byte, v.H*v.W*3)
		for c := 0; c < 3; c++ {
			s, m := PixelStd[c], PixelMean[c]
			for y := 0; y < v.H; y++ {
				for x := 0; x < v.W; x++ {
					p := min(max(float32(v.At(c, t, y, x)*s)+m, 0), 1)
					f[(y*v.W+x)*3+c] = byte(p*255 + 0.5)
				}
			}
		}
		out[t] = f
	}
	return out
}

// Tile geometry, in pixels: diffusers' defaults, and the released frames'.
const (
	tileSize    = 256
	tileOverlap = 64
)

// splitTiles is `_split_tiles`: the fewest tileSize tiles whose overlaps can
// all be at least minOverlap, with the slack handed round-robin to the
// overlaps in whole latents so every boundary stays latent-aligned.
func splitTiles(length, size, minOverlap, ratio int) (starts []int, tile int, overlaps []int) {
	if size >= length {
		return []int{0}, length, nil
	}
	n := (length + size - 1) / size
	for size*n-minOverlap*(n-1)-length < 0 {
		n++
	}
	overlaps = make([]int, n-1)
	sum := 0
	for i := range overlaps {
		overlaps[i] = minOverlap
		sum += minOverlap
	}
	remaining := size*n - sum - length
	for i := 0; i < remaining/ratio; i++ {
		overlaps[i%(n-1)] += ratio
	}
	starts = []int{0}
	for i := 0; i < n-1; i++ {
		starts = append(starts, starts[i]+size-overlaps[i])
	}
	return starts, size, overlaps
}

// Plan is where a decode's decoder calls come from: every temporal clip
// crossed with every spatial tile, in latent units.
type Plan struct {
	Clips                int   // temporal clips
	ClipTokens           int   // latent frames a clip reads: 7
	TileH, TileW         int   // a tile, in latents
	YStarts, XStarts     []int // tile origins, in latents
	YOverlaps, XOverlaps []int // tile overlaps, in pixels
	Height, Width        int   // the canvas, in pixels
	Frames               int   // pixel frames out
}

// Tiles is the number of spatial tiles a clip decodes.
func (p *Plan) Tiles() int { return len(p.YStarts) * len(p.XStarts) }

// Tokens is one decoder call's sequence: a tile-clip's latents, the
// registers and the zero token.
func (p *Plan) Tokens(c *Config) int { return p.ClipTokens*p.TileH*p.TileW + c.Registers + 1 }

// NewPlan lays out the decode of a [C, lf, lh, lw] latent.
func (c *Config) NewPlan(lf, lh, lw int) (*Plan, error) {
	nc, err := c.clips(lf)
	if err != nil {
		return nil, err
	}
	frames, _ := c.Frames(lf)
	r := c.Spatial()
	p := &Plan{Clips: nc, ClipTokens: c.ClipTokens(), Height: lh * r, Width: lw * r, Frames: frames}
	ys, th, yo := splitTiles(p.Height, tileSize, tileOverlap, r)
	xs, tw, xo := splitTiles(p.Width, tileSize, tileOverlap, r)
	p.TileH, p.TileW, p.YOverlaps, p.XOverlaps = th/r, tw/r, yo, xo
	for _, y := range ys {
		p.YStarts = append(p.YStarts, y/r)
	}
	for _, x := range xs {
		p.XStarts = append(p.XStarts, x/r)
	}
	return p, nil
}

// blendWeights are `_blend`'s ramps: b's weight i/extent, a's 1 − i/extent,
// each rounded to fp32 as torch computes them.
func blendWeights(extent int) (wa, wb []float32) {
	wa, wb = make([]float32, extent), make([]float32, extent)
	for i := range wa {
		w := float32(i) / float32(extent)
		wa[i], wb[i] = 1-w, w
	}
	return wa, wb
}

// mix is a·wa + b·wb with torch's roundings: two products, then the sum —
// the conversions keep Go from fusing them into one FMA.
func mix(a, wa, b, wb float32) float32 { return float32(a*wa) + float32(b*wb) }

// stitch is `_stitch_tiles` over one clip's decoded tiles, row-major
// (tiles[i*nx+j]). Each tile is blended with the *unblended* tile above it,
// then with the unblended tile to its left, and cropped by the overlap it
// shares with the tile below and to the right — diffusers' order, which is
// not symmetric, reproduced as it is.
func (p *Plan) stitch(tiles []*Tensor) *Tensor {
	ny, nx := len(p.YStarts), len(p.XStarts)
	t0 := tiles[0]
	out := NewTensor(t0.C, t0.T, p.Height, p.Width)
	y0 := 0
	for i := 0; i < ny; i++ {
		x0 := 0
		var rowH int
		for j := 0; j < nx; j++ {
			tile := tiles[i*nx+j]
			th, tw := tile.H, tile.W
			// Row h / column w of the result tile, before cropping.
			var eh, ew int
			var wha, whb, wwa, wwb []float32
			if i > 0 {
				eh = min(tiles[(i-1)*nx+j].H, th, p.YOverlaps[i-1])
				wha, whb = blendWeights(eh)
			}
			if j > 0 {
				ew = min(tiles[i*nx+j-1].W, tw, p.XOverlaps[j-1])
				wwa, wwb = blendWeights(ew)
			}
			keepH, keepW := th, tw
			if i < ny-1 {
				keepH -= p.YOverlaps[i]
			}
			if j < nx-1 {
				keepW -= p.XOverlaps[j]
			}
			// The vertical blend of a pixel, which the horizontal one reads
			// as its b and, for its a, the left tile's own unblended pixel.
			vert := func(c, t, h, w int) float32 {
				v := tile.At(c, t, h, w)
				if h < eh {
					up := tiles[(i-1)*nx+j]
					v = mix(up.At(c, t, up.H-eh+h, w), wha[h], v, whb[h])
				}
				return v
			}
			parallelFor(tile.C*tile.T, func(ct int) {
				c, t := ct/tile.T, ct%tile.T
				for h := 0; h < keepH; h++ {
					for w := 0; w < keepW; w++ {
						v := vert(c, t, h, w)
						if w < ew {
							left := tiles[i*nx+j-1]
							v = mix(left.At(c, t, h, left.W-ew+w), wwa[w], v, wwb[w])
						}
						out.Set(c, t, y0+h, x0+w, v)
					}
				}
			})
			x0 += keepW
			rowH = keepH
		}
		y0 += rowH
	}
	return out
}

// assemble is `_decode`'s temporal loop over the stitched clips: each clip
// contributes its first chunk (minus the encoder's pre-padding), cross-faded
// over frameOverlap frames with the previous clip's second chunk, and the
// last clip's second chunk closes the video.
func (c *Config) assemble(clips []*Tensor, frames int) *Tensor {
	c0 := clips[0]
	out := NewTensor(c0.C, frames, c0.H, c0.W)
	chunk := c.chunkTokens() * c.Temporal()
	pre, fo := c.prePadding(), c.frameOverlap()
	wa, wb := blendWeights(fo)
	plane := c0.H * c0.W
	frame := func(x *Tensor, ch, t int) []float32 { return x.Data[x.idx(ch, t, 0, 0) : x.idx(ch, t, 0, 0)+plane] }
	at := 0
	for i, clip := range clips {
		n := chunk - pre
		for f := 0; f < n; f++ {
			for ch := 0; ch < clip.C; ch++ {
				dst, src := frame(out, ch, at+f), frame(clip, ch, pre+f)
				if i > 0 && f < fo {
					// The previous clip's second chunk is its last fo frames.
					prev := clips[i-1]
					a := frame(prev, ch, prev.T-fo+f)
					for k := range dst {
						dst[k] = mix(a[k], wa[f], src[k], wb[f])
					}
				} else {
					copy(dst, src)
				}
			}
		}
		at += n
	}
	last := clips[len(clips)-1]
	for f := 0; f < fo; f++ {
		for ch := 0; ch < last.C; ch++ {
			copy(frame(out, ch, at+f), frame(last, ch, last.T-fo+f))
		}
	}
	return out
}

// parallelFor runs fn(0..n-1) over GOMAXPROCS workers.
func parallelFor(n int, fn func(i int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}
