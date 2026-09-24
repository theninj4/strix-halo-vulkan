package pixels

import "image"

// libjpeg-turbo's half of a JPEG decode, on top of Go's.
//
// Go's decoder gives the component planes (image.YCbCr) and then converts
// them with its own rules. Pillow's libjpeg-turbo converts them with
// different ones, and two of those differences are most of the gap:
//
//   - **fancy upsampling.** A 4:2:0 or 4:2:2 chroma plane is interpolated
//     with libjpeg's triangular filter (3/4 near, 1/4 far, with its
//     alternating rounding biases). Go replicates each chroma sample, which
//     on a hard colour edge is tens of levels off (V3 measured max 74 on
//     the synthetic card);
//   - **colour conversion** through libjpeg's fixed-point tables, where Go
//     uses its own rounding.
//
// Both are ported here from jdsample.c and jdcolor.c. The remaining gap is
// the IDCT: Go 1.26's is a Loeffler transform in its own fixed point, not
// jidctint's, and matching it would mean a decoder of our own
// (LLM-VISION.md V3 has the measurement).

// Fixed-point constants of jdcolor.c: FIX(x) = x * 2^16 rounded.
const (
	fixCrR = 91881  // FIX(1.40200)
	fixCbB = 116130 // FIX(1.77200)
	fixCrG = 46802  // FIX(0.71414)
	fixCbG = 22554  // FIX(0.34414)
)

var crR, cbB, crG, cbG [256]int32

func init() {
	for i := range 256 {
		x := int32(i - 128)
		crR[i] = (fixCrR*x + 1<<15) >> 16
		cbB[i] = (fixCbB*x + 1<<15) >> 16
		crG[i] = -fixCrG * x
		cbG[i] = -fixCbG*x + 1<<15
	}
}

func clamp8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// fromYCbCr converts Go's decoded planes the way libjpeg-turbo would.
func fromYCbCr(src *image.YCbCr) *RGB {
	b := src.Rect
	w, h := b.Dx(), b.Dy()
	var hs, vs int // chroma subsampling factors
	switch src.SubsampleRatio {
	case image.YCbCrSubsampleRatio444:
		hs, vs = 1, 1
	case image.YCbCrSubsampleRatio422:
		hs, vs = 2, 1
	case image.YCbCrSubsampleRatio420:
		hs, vs = 2, 2
	case image.YCbCrSubsampleRatio440:
		hs, vs = 1, 2
	default:
		// 4:1:1 and 4:1:0 are not fancy-upsampled by libjpeg either, but
		// their plain upsampling is not worth a port; take Go's.
		return nil
	}
	cw, ch := (w+hs-1)/hs, (h+vs-1)/vs
	plane := func(p []uint8) []uint8 {
		out := make([]uint8, cw*ch)
		for y := 0; y < ch; y++ {
			copy(out[y*cw:(y+1)*cw], p[y*src.CStride:])
		}
		return out
	}
	cb := upsample(plane(src.Cb), cw, ch, hs, vs, w, h)
	cr := upsample(plane(src.Cr), cw, ch, hs, vs, w, h)
	out := &RGB{W: w, H: h, Pix: make([]uint8, w*h*3)}
	for y := 0; y < h; y++ {
		row := src.Y[y*src.YStride:]
		for x := 0; x < w; x++ {
			yy := int32(row[x])
			c1, c2 := cb[y*w+x], cr[y*w+x]
			o := out.Pix[(y*w+x)*3:]
			o[0] = clamp8(yy + crR[c2])
			o[1] = clamp8(yy + (cbG[c1]+crG[c2])>>16)
			o[2] = clamp8(yy + cbB[c1])
		}
	}
	return out
}

// upsample brings a [ch, cw] chroma plane to [h, w] with jdsample.c's
// fancy filters. libjpeg only uses them when the plane is more than two
// samples wide; below that it replicates, and so does this.
func upsample(p []uint8, cw, ch, hs, vs, w, h int) []uint8 {
	if hs == 1 && vs == 1 {
		return p
	}
	fancy := cw > 2
	// Width first into [ch][cw*hs], then height into [ch*vs][cw*hs] for
	// 4:4:0; 4:2:0 is one pass over both, as h2v2_fancy_upsample is.
	ow, oh := cw*hs, ch*vs
	out := make([]uint8, ow*oh)
	row := func(y int) []uint8 {
		y = max(0, min(ch-1, y)) // libjpeg's context rows replicate the edges
		return p[y*cw : (y+1)*cw]
	}
	switch {
	case !fancy:
		for y := 0; y < oh; y++ {
			for x := 0; x < ow; x++ {
				out[y*ow+x] = p[(y/vs)*cw+x/hs]
			}
		}
	case hs == 2 && vs == 1: // h2v1_fancy_upsample
		for y := 0; y < ch; y++ {
			in, o := row(y), out[y*ow:]
			v := int32(in[0])
			o[0] = uint8(v)
			o[1] = uint8((v*3 + int32(in[1]) + 2) >> 2)
			for c := 1; c < cw-1; c++ {
				v = int32(in[c]) * 3
				o[2*c] = uint8((v + int32(in[c-1]) + 1) >> 2)
				o[2*c+1] = uint8((v + int32(in[c+1]) + 2) >> 2)
			}
			v = int32(in[cw-1])
			o[2*cw-2] = uint8((v*3 + int32(in[cw-2]) + 1) >> 2)
			o[2*cw-1] = uint8(v)
		}
	case hs == 1 && vs == 2: // h1v2_fancy_upsample (libjpeg-turbo's)
		for y := 0; y < ch; y++ {
			for v := 0; v < 2; v++ {
				in0, in1, bias := row(y), row(y-1), int32(1)
				if v == 1 {
					in1, bias = row(y+1), 2
				}
				o := out[(2*y+v)*ow:]
				for c := 0; c < cw; c++ {
					o[c] = uint8((int32(in0[c])*3 + int32(in1[c]) + bias) >> 2)
				}
			}
		}
	default: // h2v2_fancy_upsample
		for y := 0; y < ch; y++ {
			for v := 0; v < 2; v++ {
				in0, in1 := row(y), row(y-1)
				if v == 1 {
					in1 = row(y + 1)
				}
				o := out[(2*y+v)*ow:]
				sum := func(c int) int32 { return int32(in0[c])*3 + int32(in1[c]) }
				this, next := sum(0), sum(1)
				o[0] = uint8((this*4 + 8) >> 4)
				o[1] = uint8((this*3 + next + 7) >> 4)
				last := this
				this = next
				for c := 1; c < cw-1; c++ {
					next = sum(c + 1)
					o[2*c] = uint8((this*3 + last + 8) >> 4)
					o[2*c+1] = uint8((this*3 + next + 7) >> 4)
					last, this = this, next
				}
				o[2*cw-2] = uint8((this*3 + last + 8) >> 4)
				o[2*cw-1] = uint8((this*4 + 7) >> 4)
			}
		}
	}
	if ow == w && oh == h {
		return out
	}
	crop := make([]uint8, w*h) // an odd side's last interpolated sample is dropped
	for y := 0; y < h; y++ {
		copy(crop[y*w:(y+1)*w], out[y*ow:])
	}
	return crop
}
