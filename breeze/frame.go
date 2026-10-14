package breeze

// One frame on the device (B7): the depth decoder's fifteen passes, the
// backbone's step and all sixteen samples recorded once and replayed. The
// host loop (generate.go's control, GenOptions.HostSampler) read logits
// back after every pass, sampled, gathered the next code's embedding and
// re-recorded the next pass; here breeze_frame.comp does those between the
// passes, and a frame's state rides buffers the recording reads, one set a
// stream:
//
//   - fr[0..15], the frame's codes, and fr[16], the backbone's next code 0
//     (pass 0 moves it to fr[0]); the host reads fr back once a frame;
//   - the penalty's marks over codebook 0's history;
//   - sixteen uniforms a frame, the host's PCG stream in the host loop's
//     order (codebooks 1-15, then the backbone), so a seed draws the same
//     frames either way unless a draw lands within rounding of a boundary;
//   - the rows' positions (metadata, written every frame; the backbone's
//     attention key chunks are the only push constant that moves, so its
//     recording is keyed by them).
//
// **Streams share a frame** (B7, shared passes): every active utterance's
// rows ride the same passes -- two streams with CFG are four rows, which
// read the same weights as two -- with a gather and a sample a stream. The
// recordings are keyed by the composition (which streams, their CFG and
// sampler settings).
//
// A frame is cut into depthParts recordings of the depth decoder and one of
// the backbone, each its own submit under the model's lock: about 11 ms and
// 8 ms of device time, so a waiting model is never held a whole frame.

import (
	"fmt"
	"math"
	"strings"
	"time"

	"strix-halo-vulkan/safetensors"
	"strix-halo-vulkan/vk"
)

const (
	// metaPasses are the recorded depth passes' metadata slots (1..15).
	metaPasses = Codebooks - 1
	// frameSlots is a stream's fr: its frame's 16 codes, then the next code 0.
	frameSlots = 32
	frameNext  = 16
	// seenStride is a stream's penalty marks.
	seenStride = 2112
	// depthParts is how many submits the fifteen depth passes are cut into.
	depthParts = 3
)

// sampling is a request's sampler settings, which the recordings carry.
type sampling struct {
	cfg  float32
	temp float64
	topK int
	pen  float64
}

// frameStream is one utterance's part of a frame.
type frameStream struct {
	j   int       // stream slot: KV slots 2j and 2j+1, its fr, marks, uniforms
	pos []int     // its branches' backbone positions (1, or 2 with CFG)
	u   []float32 // its sixteen uniforms
	sp  sampling
}

// frameOut is a stream's frame and its next code 0.
type frameOut struct {
	frame [16]int32
	next  int32
}

func (d *Decoder) frOff(j int) uint32   { return d.aFrame + uint32(j*frameSlots) }
func (d *Decoder) seenOff(j int) uint32 { return d.aSeen + uint32(j*seenStride) }
func (d *Decoder) uOff(j int) uint32    { return d.aUniform + uint32(j*Codebooks) }
func (d *Decoder) hidOff(j int) uint32  { return d.hHid + uint32(2*j*bbHidden) }

// pair is x as fp32 hi + lo, what the sampler's float64 steps take.
func pair(x float64) (hi, lo uint32) {
	h := float32(x)
	return math.Float32bits(h), math.Float32bits(float32(x - float64(h)))
}

// NoW is dit_common.glsl's NO_W.
const NoW = 0xffffffff

// sampleDispatch samples n logits (cond at row `row`, its uncond at row+1
// when nb is 2) into stream j's fr[slot] with its uniform u.
func (d *Decoder) sampleDispatch(row, nb, n, j, slot, u int, sp sampling, penalty bool) vk.MultiDispatch {
	cond := d.aLogits + uint32(row*d.headPad)
	uncond := cond
	if nb == 2 {
		uncond += uint32(d.headPad)
	}
	pc := push{InOff: cond, KOff: uncond, Tokens: uint32(n), Scale: math.Float32bits(sp.cfg),
		Span: Codes, HeadDim: Vocab, Heads: uint32(min(sp.topK, n)),
		WOff: d.uOff(j) + uint32(u), Aux0: d.frOff(j), Aux1: uint32(slot), Aux2: NoW}
	pc.Eps, pc.LDB = pair(sp.temp)
	if penalty && sp.pen != 1 {
		pc.Aux2 = d.seenOff(j)
		pc.GemmM, pc.GemmN = pair(sp.pen)
	}
	return vk.MultiDispatch{Pipeline: d.pipes["sample"], GroupsX: 1, GroupsY: 1, PushConstants: pc.bytes()}
}

// branches is the streams' rows in order and each stream's first.
func branches(ss []frameStream) (total int, first []int) {
	for _, s := range ss {
		first = append(first, total)
		total += len(s.pos)
	}
	return total, first
}

// depthPassRows is depth pass p's rows: on pass 0 every branch's hidden row
// (position 0) and then every branch's code-0 row (position 1); after it,
// every branch's codebook-p row at position p+1.
func depthPassRows(p int, ss []frameStream) (rows []Row, heads []int) {
	if p == 0 {
		for _, s := range ss {
			for b := range s.pos {
				rows, heads = append(rows, Row{2*s.j + b, 0}), append(heads, -1)
			}
		}
	}
	for _, s := range ss {
		for b := range s.pos {
			rows, heads = append(rows, Row{2*s.j + b, p + 1}), append(heads, p)
		}
	}
	return rows, heads
}

// depthPassGraph is depth pass p (codebook p+1's logits) of a frame: each
// stream's input gather, the decoder over every row, each stream's sample
// into its fr[p+1].
func (d *Decoder) depthPassGraph(p int, ss []frameStream) ([]vk.MultiDispatch, error) {
	total, first := branches(ss)
	codeRow0 := 0
	if p == 0 {
		codeRow0 = total
	}
	var g []vk.MultiDispatch
	for i, s := range ss {
		nb := len(s.pos)
		gp := push{LDA: uint32(d.ldaIn), BOff: d.embOff, Aux0: d.frOff(s.j), Heads: uint32(nb), Aux1: uint32(p),
			KOff: d.hIn + uint32((codeRow0+first[i])*d.ldaIn)}
		if p == 0 {
			gp.Tokens, gp.Aux2 = uint32(nb), 1
			gp.InOff, gp.OutOff = d.hidOff(s.j), d.hIn+uint32(first[i]*d.ldaIn)
		}
		g = append(g, vk.MultiDispatch{Pipeline: d.pipes["gather"], GroupsX: gp.Tokens + gp.Heads, GroupsY: 1,
			PushConstants: gp.bytes()})
	}
	rows, heads := depthPassRows(p, ss)
	dg, err := d.depthGraph(rows, heads, 1+p)
	if err != nil {
		return nil, err
	}
	g = append(g, dg...)
	for i, s := range ss {
		g = append(g, d.sampleDispatch(codeRow0+first[i], len(s.pos), Vocab, s.j, p+1, p, s.sp, false))
	}
	return g, nil
}

// backboneRows is the backbone step's rows: every branch at its position.
func backboneRows(ss []frameStream) []Row {
	var rows []Row
	for _, s := range ss {
		for b, p := range s.pos {
			rows = append(rows, Row{2*s.j + b, p})
		}
	}
	return rows
}

// backboneStepGraph is the backbone's step on each stream's frame: the
// summed embeddings, the stack over every row, the head, each stream's
// sample into its fr[16] (marked for the penalty), and the hidden rows kept
// for the next frame's depth pass.
func (d *Decoder) backboneStepGraph(ss []frameStream) ([]vk.MultiDispatch, error) {
	total, first := branches(ss)
	var g []vk.MultiDispatch
	for i, s := range ss {
		g = append(g, vk.MultiDispatch{Pipeline: d.pipes["embed"], GroupsX: bbHidden / 256, GroupsY: 1,
			PushConstants: push{OutOff: d.bb.aX + uint32(first[i]*bbHidden), Tokens: uint32(len(s.pos)),
				Aux0: d.frOff(s.j), BOff: d.embOff}.bytes()})
	}
	sg, err := d.stackGraph(d.bb, backboneRows(ss), 0)
	if err != nil {
		return nil, err
	}
	g = append(g, sg...)
	g = append(g, d.proj(headBank, 1, d.bb.hA, d.bb.ldaDim, d.aLogits, d.lmHead, d.headPad, bbHidden, total)...)
	for i, s := range ss {
		g = append(g, d.sampleDispatch(first[i], len(s.pos), Vocab+1, s.j, frameNext, Codebooks-1, s.sp, true))
	}
	for i, s := range ss {
		g = append(g, vk.MultiDispatch{Pipeline: d.pipes["copyh"], GroupsX: uint32(len(s.pos)), GroupsY: 1,
			PushConstants: push{InOff: d.bb.hA + uint32(first[i]*d.bb.ldaDim), LDA: uint32(d.bb.ldaDim),
				OutOff: d.hidOff(s.j)}.bytes()})
	}
	return g, nil
}

// composition keys a recording: the part, the streams' slots, branch counts
// and sampler settings, and (for the backbone) its key chunks.
func composition(part, chunks int, ss []frameStream) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d/%d", part, chunks)
	for _, s := range ss {
		fmt.Fprintf(&b, ";%d,%d,%v", s.j, len(s.pos), s.sp)
	}
	return b.String()
}

// recording is the cached recording for key, built by graph on a miss.
func (d *Decoder) recording(key string, graph func() ([]vk.MultiDispatch, error)) (*vk.Prerecorded, error) {
	if r := d.recs[key]; r != nil {
		return r, nil
	}
	if d.recs == nil || len(d.recs) > 512 {
		for _, r := range d.recs {
			r.Destroy()
		}
		d.recs = map[string]*vk.Prerecorded{}
	}
	g, err := graph()
	if err != nil {
		return nil, err
	}
	r, err := vk.NewPrerecorded(g, true, false)
	if err != nil {
		return nil, err
	}
	d.recs[key] = r
	return r, nil
}

// startFrames seeds stream j's device frame: the prefill's last hidden row
// per branch, the first code 0, and the penalty's history holding it.
func (d *Decoder) startFrames(j int, hidden [][]float32, tok int32) {
	h := make([]uint16, len(hidden)*bbHidden)
	for b, row := range hidden {
		for i, v := range row {
			h[b*bbHidden+i] = safetensors.F32ToF16(v)
		}
	}
	d.hbuf.WriteUint16At(int(d.hidOff(j)), h)
	d.abuf.ZeroFloat32At(int(d.seenOff(j)), seenStride)
	d.abuf.WriteUint32At(int(d.seenOff(j))+int(tok), []uint32{1})
	d.abuf.WriteUint32At(int(d.frOff(j))+frameNext, []uint32{uint32(tok)})
}

// frame runs one frame of every stream in ss. run wraps each submit (the
// model's lock); the host's writes before a submit are made inside it.
func (d *Decoder) frame(ss []frameStream, run func(func() error) error) (out []frameOut, took time.Duration, err error) {
	total, _ := branches(ss)
	if 2*total > MaxRows {
		return nil, 0, fmt.Errorf("breeze: a frame of %d branches; at most %d", total, MaxRows/2)
	}
	maxKeys := 0
	for _, s := range ss {
		for _, p := range s.pos {
			maxKeys = max(maxKeys, p+1)
		}
	}
	chunks, _ := keyChunks(total, d.bb.nq, maxKeys)
	per := (metaPasses + depthParts - 1) / depthParts
	for part := 0; part <= depthParts; part++ {
		err = run(func() error {
			var rec *vk.Prerecorded
			var err error
			if part < depthParts {
				if part == 0 {
					for _, s := range ss {
						d.abuf.WriteFloat32At(int(d.uOff(s.j)), s.u)
					}
				}
				lo, hi := part*per, min(metaPasses, (part+1)*per)
				for p := lo; p < hi; p++ {
					rows, _ := depthPassRows(p, ss)
					if _, _, err = d.writeMeta(d.dd, rows, 1+p); err != nil {
						return err
					}
				}
				rec, err = d.recording(composition(part, 0, ss), func() ([]vk.MultiDispatch, error) {
					var g []vk.MultiDispatch
					for p := lo; p < hi; p++ {
						pg, err := d.depthPassGraph(p, ss)
						if err != nil {
							return nil, err
						}
						g = append(g, pg...)
					}
					return g, nil
				})
			} else {
				if _, _, err = d.writeMeta(d.bb, backboneRows(ss), 0); err != nil {
					return err
				}
				rec, err = d.recording(composition(part, chunks, ss), func() ([]vk.MultiDispatch, error) {
					return d.backboneStepGraph(ss)
				})
			}
			if err != nil {
				return err
			}
			dt, _, err := rec.Submit()
			took += dt
			if err != nil || part < depthParts {
				return err
			}
			out = make([]frameOut, len(ss))
			for i, s := range ss {
				fr := d.abuf.ReadUint32At(int(d.frOff(s.j)), frameNext+1)
				for k := range out[i].frame {
					out[i].frame[k] = int32(fr[k])
				}
				out[i].next = int32(fr[frameNext])
			}
			return nil
		})
		if err != nil {
			return nil, took, err
		}
	}
	return out, took, nil
}
