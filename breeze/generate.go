package breeze

// The generation loop (B4): upstream's FastBreezeStreamingRuntime.
// iter_audio_chunks, step for step. A frame is codebook 0 from the
// backbone's head, then codebooks 1-15 from the depth decoder fed the
// backbone's normed hidden state; with CFG the prompt and its negative run as
// two slots of every pass, the scale is applied to the backbone's logits and
// to every codebook's, and one sampled token is shared by both rows.
// Frames go to the codec as they are made, a chunk at a time.

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"strix-halo-vulkan/gemma4"
	"strix-halo-vulkan/vk"
)

// Model is the whole of Breeze-TTS-2, resident.
type Model struct {
	Tok   *gemma4.Tokenizer
	Enc   *Encoder
	Dec   *Decoder
	Codec *Codec
}

// ModelOptions size the device state.
type ModelOptions struct {
	Decoder Options
	// EncoderTokens is the longest text segment.
	EncoderTokens int
	// CodecChunk is the most frames the codec decodes at once.
	CodecChunk int
}

// DefaultModelOptions is upstream's bounds: 2048 backbone positions (prompt
// and frames), segments of up to 1024 tokens.
func DefaultModelOptions() ModelOptions {
	return ModelOptions{Decoder: DefaultOptions(), EncoderTokens: 1024, CodecChunk: 32}
}

// LoadModel stages every part from the checkpoint directory.
func LoadModel(dev *vk.Device, dir string, o ModelOptions) (*Model, error) {
	m := &Model{}
	var err error
	if m.Tok, err = gemma4.LoadTokenizer(dir); err != nil {
		return nil, err
	}
	if m.Enc, err = LoadEncoder(dev, dir, o.EncoderTokens); err != nil {
		return nil, err
	}
	if m.Dec, err = Load(dev, dir, o.Decoder); err != nil {
		m.Destroy()
		return nil, err
	}
	if m.Codec, err = LoadCodec(dev, dir, o.CodecChunk, o.Decoder.MaxLen); err != nil {
		m.Destroy()
		return nil, err
	}
	return m, nil
}

// Destroy releases every part.
func (m *Model) Destroy() {
	if m.Enc != nil {
		m.Enc.Destroy()
	}
	if m.Dec != nil {
		m.Dec.Destroy()
	}
	if m.Codec != nil {
		m.Codec.Destroy()
	}
}

// GenOptions are the sampling settings; the zero value of each field takes
// upstream's default.
type GenOptions struct {
	// CFG is the guidance scale. Upstream's CLI defaults to 1 (none); its
	// README asks for 4 with an instruction, which DefaultCFG returns.
	CFG         float64
	Temperature float64 // 0.9
	TopK        int     // 50
	RepPenalty  float64 // 1.1, over codebook 0's history
	MaxFrames   int     // 1500 (120 s)
	Seed        uint64
	// Chunk is the frames handed to the codec at once (2, upstream's eager
	// chunk; at most the codec's Chunk).
	Chunk int
	// Lock, when set, wraps every use of the device (a server's device
	// lock): one encoder segment, one prefill pass, one backbone step, one
	// depth pass, one codec chunk. Between them the device is free, so a
	// long utterance does not hold other models off it.
	Lock func(func() error) error
}

// DefaultCFG is the README's scale for a template with a negative prompt
// (an instruction), and 1 otherwise.
func DefaultCFG(t Template) float64 {
	if t == TTSInstruction || t == RefEdit {
		return 4
	}
	return 1
}

func (o *GenOptions) defaults(t Template) {
	if o.CFG == 0 {
		o.CFG = DefaultCFG(t)
	}
	if o.Temperature == 0 {
		o.Temperature = 0.9
	}
	if o.TopK == 0 {
		o.TopK = 50
	}
	if o.RepPenalty == 0 {
		o.RepPenalty = 1.1
	}
	if o.MaxFrames == 0 {
		o.MaxFrames = 1500
	}
	if o.Chunk == 0 {
		o.Chunk = 2
	}
}

// Stats is where a generation's time went.
type Stats struct {
	Frames                           int
	Prompt                           int // backbone prompt rows (the longest branch)
	Encode, Prefill, Backbone, Depth time.Duration
	Codec, Sample                    time.Duration
	FirstAudio, Total                time.Duration
}

// Generate synthesises a request, handing the audio to emit chunk by chunk
// (24 kHz mono, [-1, 1]).
func (m *Model) Generate(ctx context.Context, req *Request, o GenOptions, emit func([]float32) error) (*Stats, error) {
	t0 := time.Now()
	do := o.Lock
	if do == nil {
		do = func(f func() error) error { return f() }
	}
	p, err := BuildPrompt(m.Tok, req)
	if err != nil {
		return nil, err
	}
	o.defaults(p.Template)
	if o.Chunk > m.Codec.Chunk {
		o.Chunk = m.Codec.Chunk
	}
	branches := []*Branch{&p.Cond}
	if o.CFG != 1 {
		if p.Uncond == nil {
			return nil, fmt.Errorf("breeze: a guidance scale of %g needs a template with a negative prompt (%s has none)", o.CFG, p.Template)
		}
		branches = append(branches, p.Uncond)
	}
	st := &Stats{}
	d := m.Dec

	// The prompts, prefilled one slot each: [L][2048] embeddings.
	lens := make([]int, len(branches))
	hidden := make([][]float32, len(branches))
	logits := make([][]float32, len(branches))
	for b, br := range branches {
		x, dt, err := m.embedPrompt(br, do)
		if err != nil {
			return nil, err
		}
		st.Encode += dt
		L := len(br.IDs)
		lens[b] = L
		st.Prompt = max(st.Prompt, L)
		if L >= d.o.MaxLen {
			return nil, fmt.Errorf("breeze: a prompt of %d positions; the backbone holds %d", L, d.o.MaxLen)
		}
		for at := 0; at < L; at += d.o.Rows {
			n := min(d.o.Rows, L-at)
			rows := make([]Row, n)
			for i := range rows {
				rows[i] = Row{b, at + i}
			}
			want := 0
			if at+n == L {
				want = 1
			}
			var h, lg [][]float32
			var dt time.Duration
			err := do(func() (err error) {
				h, lg, dt, err = d.Backbone(x[at*bbHidden:(at+n)*bbHidden], rows, want)
				return err
			})
			if err != nil {
				return nil, err
			}
			st.Prefill += dt
			if want > 0 {
				hidden[b], logits[b] = h[n-1], lg[0]
			}
		}
	}

	rng := rand.New(rand.NewPCG(o.Seed, 0x6272656531)) // "bree1"
	s := &sampler{temp: o.Temperature, topK: o.TopK, rng: rng}
	var history []int32
	cfg := func(lg [][]float32, n int) []float32 {
		out := make([]float32, n)
		if len(lg) == 1 {
			copy(out, lg[0][:n])
			return out
		}
		for i := range out {
			out[i] = lg[1][i] + float32(o.CFG)*(lg[0][i]-lg[1][i])
		}
		return out
	}
	ts := time.Now()
	tok := s.sample(cfg(logits, Vocab+1), nil, 1, Codes, Vocab)
	st.Sample += time.Since(ts)

	m.Codec.Reset()
	var pending [][16]int32
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		var wav []float32
		var dt time.Duration
		err := do(func() (err error) {
			wav, dt, err = m.Codec.Decode(pending)
			return err
		})
		st.Codec += dt
		if err != nil {
			return err
		}
		pending = pending[:0]
		if st.FirstAudio == 0 {
			st.FirstAudio = time.Since(t0)
		}
		return emit(wav)
	}

	nb := len(branches)
	in := make([]float32, 2*nb*bbHidden)
	x := make([]float32, nb*bbHidden)
	for step := 0; step < o.MaxFrames && tok != CodebookEOS; step++ {
		if err := ctx.Err(); err != nil {
			return st, err
		}
		if lens[0]+step >= d.o.MaxLen-1 {
			break
		}
		// The depth decoder: rows (slot, 0) are the backbone's hidden
		// states and (slot, 1) codebook 0's row, then one row a slot a step.
		var frame [16]int32
		frame[0] = tok
		rows := make([]Row, 0, 2*nb)
		heads := make([]int, 0, 2*nb)
		for b := 0; b < nb; b++ {
			copy(in[b*bbHidden:], hidden[b])
			rows, heads = append(rows, Row{b, 0}), append(heads, -1)
		}
		for b := 0; b < nb; b++ {
			d.codeEmbed(in[(nb+b)*bbHidden:(nb+b+1)*bbHidden], 0, tok)
			rows, heads = append(rows, Row{b, 1}), append(heads, 0)
		}
		var lg [][]float32
		var dt time.Duration
		err := do(func() (err error) {
			lg, dt, err = d.Depth(in, rows, heads)
			return err
		})
		if err != nil {
			return st, err
		}
		st.Depth += dt
		for k := 1; k < Codebooks; k++ {
			ts := time.Now()
			frame[k] = s.sample(cfg(lg[len(lg)-nb:], Vocab), nil, 1, Codes, Vocab)
			st.Sample += time.Since(ts)
			if k == Codebooks-1 {
				break
			}
			rows, heads = rows[:0], heads[:0]
			for b := 0; b < nb; b++ {
				d.codeEmbed(in[b*bbHidden:(b+1)*bbHidden], k, frame[k])
				rows, heads = append(rows, Row{b, k + 1}), append(heads, k)
			}
			if err = do(func() (err error) {
				lg, dt, err = d.Depth(in[:nb*bbHidden], rows, heads)
				return err
			}); err != nil {
				return st, err
			}
			st.Depth += dt
		}
		st.Frames++
		pending = append(pending, frame)
		if len(pending) >= o.Chunk {
			if err := flush(); err != nil {
				return st, err
			}
		}

		// The backbone's next step: the frame's summed code rows.
		history = append(history, tok)
		brow := make([]Row, nb)
		for b := 0; b < nb; b++ {
			d.frameEmbed(x[b*bbHidden:(b+1)*bbHidden], frame)
			brow[b] = Row{b, lens[b] + step}
		}
		var h, lgs [][]float32
		err = do(func() (err error) {
			h, lgs, dt, err = d.Backbone(x, brow, nb)
			return err
		})
		if err != nil {
			return st, err
		}
		st.Backbone += dt
		copy(hidden, h)
		ts := time.Now()
		tok = s.sample(cfg(lgs, Vocab+1), history, o.RepPenalty, Codes, Vocab)
		st.Sample += time.Since(ts)
	}
	if err := flush(); err != nil {
		return st, err
	}
	st.Total = time.Since(t0)
	return st, nil
}

// embedPrompt is a branch's backbone input: each text segment through the
// encoder and the projection, each reference frame's summed code rows, and
// <|audio_eos|> as the all-zero frame's (codebook_eos_token_id).
func (m *Model) embedPrompt(br *Branch, do func(func() error) error) ([]float32, time.Duration, error) {
	L := len(br.IDs)
	x := make([]float32, L*bbHidden)
	var took time.Duration
	segs := br.TextSegments()
	si, frame := 0, 0
	for i := 0; i < L; {
		if br.TextMask[i] {
			var out []float32
			var dt time.Duration
			err := do(func() (err error) {
				out, dt, err = m.Enc.Encode(segs[si], nil)
				return err
			})
			if err != nil {
				return nil, took, err
			}
			took += dt
			copy(x[i*bbHidden:], out)
			i += len(segs[si])
			si++
			continue
		}
		row := x[i*bbHidden : (i+1)*bbHidden]
		switch br.IDs[i] {
		case AudioTok:
			m.Dec.frameEmbed(row, br.Frames[frame])
			frame++
		case AudioEOS:
			m.Dec.frameEmbed(row, [16]int32{})
		default:
			return nil, took, fmt.Errorf("breeze: id %d at a non-text position", br.IDs[i])
		}
		i++
	}
	return x, took, nil
}

// sampler is upstream's sample_logits: repetition penalty over the unique
// history, the reserved ids [lo, hi) suppressed, temperature, top-k (every
// logit at or above the k-th largest kept), softmax, one multinomial draw.
type sampler struct {
	temp float64
	topK int
	rng  *rand.Rand
	buf  []float32
}

// kthLargest is the k-th largest of v (torch.topk's last value) and the
// largest, by a size-k min-heap: O(n log k), where a sort of a codebook's
// 2051 logits sixteen times a frame was 3 ms of host time a frame.
func kthLargest(v []float32, k int, buf *[]float32) (kth, top float32) {
	h := (*buf)[:0]
	down := func(i int) {
		for {
			l, m := 2*i+1, i
			if l < len(h) && h[l] < h[m] {
				m = l
			}
			if r := l + 1; r < len(h) && h[r] < h[m] {
				m = r
			}
			if m == i {
				return
			}
			h[i], h[m] = h[m], h[i]
			i = m
		}
	}
	top = float32(math.Inf(-1))
	for _, x := range v {
		top = max(top, x)
		if len(h) < k {
			h = append(h, x)
			for i := len(h) - 1; i > 0 && h[(i-1)/2] > h[i]; i = (i - 1) / 2 {
				h[i], h[(i-1)/2] = h[(i-1)/2], h[i]
			}
			continue
		}
		if x > h[0] {
			h[0] = x
			down(0)
		}
	}
	*buf = h
	return h[0], top
}

func (s *sampler) sample(lg []float32, history []int32, penalty float64, lo, hi int) int32 {
	if penalty != 1 {
		seen := map[int32]bool{}
		for _, t := range history {
			if seen[t] {
				continue
			}
			seen[t] = true
			if v := lg[t]; v > 0 {
				lg[t] = float32(float64(v) / penalty)
			} else {
				lg[t] = float32(float64(v) * penalty)
			}
		}
	}
	for i := lo; i < hi && i < len(lg); i++ {
		lg[i] = float32(math.Inf(-1))
	}
	for i := range lg {
		lg[i] = float32(float64(lg[i]) / s.temp)
	}
	thr, mx := kthLargest(lg, min(s.topK, len(lg)), &s.buf)
	var z float64
	for _, v := range lg {
		if v >= thr {
			z += math.Exp(float64(v - mx))
		}
	}
	u := s.rng.Float64() * z
	last := int32(0)
	for i, v := range lg {
		if v < thr {
			continue
		}
		last = int32(i)
		u -= math.Exp(float64(v - mx))
		if u <= 0 {
			return int32(i)
		}
	}
	return last
}

// Bytes is the device memory the model holds: weights, arenas, caches.
func (m *Model) Bytes() int {
	n := 0
	for _, b := range []*vk.Buffer{m.Enc.wbuf, m.Enc.abuf, m.Enc.hbuf, m.Enc.bank,
		m.Dec.wbuf, m.Dec.abuf, m.Dec.hbuf, m.Dec.banks[0], m.Dec.banks[1],
		m.Codec.wbuf, m.Codec.abuf, m.Codec.hbuf, m.Codec.bank} {
		if b != nil {
			n += b.Size()
		}
	}
	return n
}
