package gemma4

// A decisions request on the device (R6): prompts, the shared prefix, one
// packed pass, the label readout.

import (
	"fmt"
	"strings"

	"strix-halo-vulkan/decide"
)

// Usage is what a request cost, in v1's terms.
type Usage struct {
	InputTokens  int // the shared prefix once plus every question's suffix
	OutputTokens int // the number of questions
}

// Codebook is v1's label vocabulary past 26 options for this tokenizer.
func (t *Tokenizer) Codebook() []string {
	return decide.Codebook(func(code string) (int, bool) {
		ids := t.Encode(code)
		if len(ids) != 1 {
			return 0, false
		}
		return int(ids[0]), t.Decode(ids, false) == code
	})
}

// pieceLen is how many bytes of the prompt text token id stands for.
func (t *Tokenizer) pieceLen(id int32) int {
	p := t.pieces[id]
	if _, ok := byteToken(p); ok {
		return 1
	}
	return len(strings.ReplaceAll(p, "▁", " "))
}

// prompt is one question ready to run.
type prompt struct {
	ids    []int32
	labels []int32 // each label's token, in option order
	cap    int     // tokens wholly before "QUESTION:"
}

// Answer runs one decisions request in a single pass and resolves every
// question at calibration temperature temp.
func (g *GPU) Answer(req *decide.Request, codebook []string, temp float64) ([]decide.Answer, Usage, error) {
	ro, err := g.Readout(req, codebook)
	if err != nil {
		return nil, ro.Usage, err
	}
	answers := make([]decide.Answer, len(req.Questions))
	for i := range req.Questions {
		a, err := decide.Resolve(&req.Questions[i], ro.Logits[i], temp)
		if err != nil {
			return nil, ro.Usage, err
		}
		answers[i] = a
	}
	return answers, ro.Usage, nil
}

// Readout is one request's label logits, before the temperature: a row a
// question (order-averaged where asked), and the prompt each question ran
// as, for an off-device reference to re-run.
type Readout struct {
	Logits  [][]float32
	Prompts [][]int32 // each question's whole prompt, as sent
	Labels  [][]int32 // each question's label tokens
	Usage   Usage
}

// Readout runs one decisions request in a single pass.
func (g *GPU) Readout(req *decide.Request, codebook []string) (Readout, error) {
	ros, errs := g.ReadoutBatch([]*decide.Request{req}, codebook)
	return ros[0], errs[0]
}

// Plan is a request checked and cut into its pass segment, ready to batch.
type Plan struct {
	req  *decide.Request
	pl   *requestPlan
	pics []*Picture
}

// Segment is the share of a pass the plan takes.
func (p *Plan) Segment() Segment {
	sg := Segment{State: p.pl.prompts[0].ids[:p.pl.pref], Branches: p.pl.branches}
	for _, pic := range p.pics {
		sg.Soft = append(sg.Soft, pic.Feats)
	}
	return sg
}

// Rows is the pass rows the plan takes.
func (p *Plan) Rows() int { return p.pl.use.InputTokens }

// PlanRequest checks a request and tokenises it: everything before the
// device, so a refusal costs no pass. A request with images goes through
// PlanImages instead.
func (g *GPU) PlanRequest(req *decide.Request, codebook []string) (*Plan, error) {
	return g.PlanImages(req, codebook, nil)
}

// PlanImages is PlanRequest for a request whose images the caller has read
// and patchified (pics, in the request's order; Picture.Feats filled by
// EncodePictures before the pass). They go into every question's user turn
// ahead of the text, as decisions v1 renders `images`.
func (g *GPU) PlanImages(req *decide.Request, codebook []string, pics []*Picture) (*Plan, error) {
	if len(req.Images) > 0 && g.Vis == nil {
		return nil, &decide.Error{Message: "this server was started without vision; images are not accepted",
			Param: "images", Code: "vision_disabled"}
	}
	if len(pics) != len(req.Images) {
		return nil, fmt.Errorf("gemma4: %d images read for a request with %d", len(pics), len(req.Images))
	}
	if req.Thinking {
		return nil, &decide.Error{Message: "thinking is not supported by this server yet",
			Param: "thinking", Code: "decisions_thinking_not_supported"}
	}
	var soft []int
	for _, p := range pics {
		soft = append(soft, p.Patches.Soft)
	}
	pl, err := g.Tok.plan(req, codebook, ImageText(soft))
	if err != nil {
		return nil, err
	}
	plan := &Plan{req: req, pl: pl, pics: pics}
	if !g.Fits([]Segment{plan.Segment()}) {
		return nil, &decide.Error{Message: fmt.Sprintf("the request is %d tokens, longer than the model's pass of %d",
			pl.use.InputTokens, g.rows), Param: "state", Code: "invalid_decisions_request"}
	}
	return plan, nil
}

// Pictures are the plan's images, for EncodePictures.
func (p *Plan) Pictures() []*Picture { return p.pics }

// RunPlans answers planned requests in one pass (R8): they share its weight
// reads. The caller packs them so they fit (Fits on their segments).
func (g *GPU) RunPlans(plans []*Plan) ([]Readout, error) {
	segs := make([]Segment, len(plans))
	for i, p := range plans {
		segs[i] = p.Segment()
	}
	pass, err := g.NewBatch(segs)
	if err != nil {
		return nil, err
	}
	if _, err := g.Forward(pass, 0); err != nil {
		return nil, err
	}
	out := make([]Readout, len(plans))
	at := 0 // the pass's readouts, in segment then branch order
	for i, p := range plans {
		pl := p.pl
		rows := make([][]float32, len(pl.prompts))
		for k, pr := range pl.prompts {
			rows[k] = g.LabelLogits(g.FinalNorm(g.Residual(pass.readouts[at+k])), pr.labels)
		}
		at += len(pl.prompts)
		mirrorOf := map[int]int{}
		for k, rd := range pl.reads {
			if rd.mirror {
				mirrorOf[rd.q] = k
			}
		}
		ro := Readout{Usage: pl.use}
		for q := range p.req.Questions {
			z := rows[q]
			if k, ok := mirrorOf[q]; ok {
				z = decide.AverageOrders(z, rows[k])
			}
			ro.Logits = append(ro.Logits, z)
			ro.Prompts = append(ro.Prompts, pl.prompts[q].ids)
			ro.Labels = append(ro.Labels, pl.prompts[q].labels)
		}
		out[i] = ro
	}
	return out, nil
}

// ReadoutBatch plans and runs several requests in one pass; a request that
// fails its plan gets its error and the rest still run.
func (g *GPU) ReadoutBatch(reqs []*decide.Request, codebook []string) ([]Readout, []error) {
	ros := make([]Readout, len(reqs))
	errs := make([]error, len(reqs))
	var plans []*Plan
	var idx []int
	for i, r := range reqs {
		p, err := g.PlanRequest(r, codebook)
		if err != nil {
			errs[i] = err
			continue
		}
		plans, idx = append(plans, p), append(idx, i)
	}
	if len(plans) == 0 {
		return ros, errs
	}
	out, err := g.RunPlans(plans)
	for k, i := range idx {
		if err != nil {
			errs[i] = err
			continue
		}
		ros[i] = out[k]
	}
	return ros, errs
}

// mirrored is a choice or noul question with its options in reverse order:
// a noul's mirror shows true as A and false as B. Only its prompt is used;
// the answer is resolved on the question as sent (AverageOrders).
func mirrored(q decide.Question) decide.Question {
	m := q
	n := len(q.Keys)
	m.Keys, m.Texts, m.Values = make([]string, n), make([]string, n), make([]decide.Value, n)
	for i := range n {
		m.Keys[i], m.Texts[i], m.Values[i] = q.Keys[n-1-i], q.Texts[n-1-i], q.Values[n-1-i]
	}
	return m
}

// reading is one row of a request's readout: a question as sent, or with
// order averaging a choice or noul question again with its options reversed.
type reading struct {
	q      int
	mirror bool
}

// requestPlan is a request tokenised and cut into the shared prefix and one
// branch a reading.
type requestPlan struct {
	prompts  []prompt
	reads    []reading
	pref     int
	branches [][]int32
	use      Usage
}

// plan renders and tokenises every reading and finds the shared prefix:
// the longest common token prefix of the prompts, capped at the tokens
// wholly before "QUESTION:" and at one token less than the shortest prompt
// (decisions v1's rule; merges can cross the state/question join, so it is
// found on tokens).
func (t *Tokenizer) plan(req *decide.Request, codebook []string, images string) (*requestPlan, error) {
	var reads []reading
	for i := range req.Questions {
		reads = append(reads, reading{i, false})
	}
	if req.OrderAveraging {
		for i, q := range req.Questions {
			if q.Type != decide.Score {
				reads = append(reads, reading{i, true})
			}
		}
	}
	prompts := make([]prompt, len(reads))
	for k, rd := range reads {
		q := req.Questions[rd.q]
		if rd.mirror {
			q = mirrored(q)
		}
		r, err := decide.Render(&q, codebook)
		if err != nil {
			return nil, err
		}
		user := images + req.StateText + r.Branch
		text := DecisionPrompt(r.System, user)
		p := prompt{ids: t.Encode(text)}
		// The prompt ends in special tokens, so a label after it is a
		// segment of its own: one token in context is one token alone.
		for _, l := range r.Labels {
			lt := t.Encode(l)
			if len(lt) != 1 {
				return nil, &decide.Error{Message: fmt.Sprintf("option label %q is not one token for this model", l),
					Param: "questions", Code: "invalid_decisions_request"}
			}
			p.labels = append(p.labels, lt[0])
		}
		before := len(turnSystem) + len(r.System) + len(turnUser) + len(images) + len(req.StateText)
		at := 0
		for _, id := range p.ids {
			if at+t.pieceLen(id) > before {
				break
			}
			at += t.pieceLen(id)
			p.cap++
		}
		prompts[k] = p
	}
	// The shared prefix: the longest common token prefix, capped before
	// QUESTION: and at one token less than the shortest prompt.
	pref := prompts[0].cap
	for _, p := range prompts {
		pref = min(pref, p.cap, len(p.ids)-1)
		n := 0
		for n < pref && p.ids[n] == prompts[0].ids[n] {
			n++
		}
		pref = n
	}
	branches := make([][]int32, len(prompts))
	use := Usage{InputTokens: pref, OutputTokens: len(req.Questions)}
	for k, p := range prompts {
		branches[k] = p.ids[pref:]
		use.InputTokens += len(branches[k])
	}
	return &requestPlan{prompts: prompts, reads: reads, pref: pref, branches: branches, use: use}, nil
}

// PassOf is the packed pass a request runs as, for the profiler.
func (g *GPU) PassOf(req *decide.Request, codebook []string) (*Pass, error) {
	pl, err := g.Tok.plan(req, codebook, "")
	if err != nil {
		return nil, err
	}
	return g.NewPass(pl.prompts[0].ids[:pl.pref], pl.branches)
}
