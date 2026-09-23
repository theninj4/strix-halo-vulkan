package backend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/util"
	"strix-halo-vulkan/vk"
	"strix-halo-vulkan/zimage/tokenizer"
)

// LLMOptions is what cmd/serve's flags come to.
type LLMOptions struct {
	// Model is the GGUF checkpoint: a directory holding exactly one, or any
	// one of its shards.
	Model string
	// Device is required. There is no host path for this model -- 48 layers
	// and 512 experts in Go on a CPU is not a fallback, it is a different
	// project -- so a nil device is an error rather than a slower run.
	Device *Device
	// Context is the attention cache's cell count, and so the longest
	// conversation this process can hold: prompt plus completion. It is a
	// residency decision made once at load, not per request.
	Context int
	// Batch is how many tokens the arenas hold, which is the chunk a prompt
	// is prefilled in.
	//
	// **It was 512 — llama.cpp's own best ubatch on this part, and what L1's
	// baseline is against — and 512 is the worst rung this graph has** (P11).
	// llama.cpp plateaus at its ubatch; this one does not, because the MoE's
	// arithmetic intensity is the *routing's*: at 512 tokens 274 of 512
	// experts are touched to serve 5120 rows, every one of them unpacked
	// whole, and the schedule pads 5120 rows out to 11 904. Widening the
	// chunk spreads that same unpack over more rows and costs only arenas.
	// Measured at 48 layers and ctx 4096, two runs each:
	//
	//	512    1.16 GB of arenas    667.4 tok/s
	//	2048   2.28 GB             1093.3   1.64x
	//	4096   3.78 GB             1237.0   1.85x
	//
	// The default is **4096** (P12), and since P16 it is **free**. P12-7
	// measured it through the server as a trade — 1.12x on prefill for 0.92x
	// on decode, breaking even at ~150 generated tokens — and that decode
	// cost was a bug: the DeltaNet and attention input ports padded a decode
	// step's one row with zeros out to the whole arena, once a layer. With
	// the port padded to its row block, the same measurement (four distinct
	// ~3.9-4.3k-token prompts at ctx 8192, 256 generated, two passes
	// agreeing to 0.1%) reads:
	//
	//	2048   1052.2 tok/s prefill    34.19 tok/s decode
	//	4096   1198.5          1.14x   34.20
	//	8192   1232.5          1.17x   34.16
	//
	// So a wider batch is now a question of memory and nothing else: 8192
	// helps only prompts longer than 4096 tokens (946 against 886 tok/s at
	// 128k, P14), costs ~3 GB more arena than 4096, and lowers the
	// maxStorageBufferRange cap on the context. Arenas, at ctx 4096: 2.28 GB
	// at 2048 against 3.78 at 4096.
	Batch int
	// MaxTokens is what a request that names no budget gets. A request that
	// names one still cannot run past the context.
	MaxTokens int
	// Layers truncates the model to its first N layers. It is not a model,
	// it is how the server is started in seconds instead of a minute while
	// the HTTP side is being worked on (LLM.md's 4-layer prefix).
	Layers int
	// ID is the model id this backend answers to in /v1/models.
	ID string
	// Slots is how many conversations the graph holds at once
	// (CONCURRENCY.md C1/C2). Each is its own KV cache, DeltaNet state and
	// rings — ~27.8 KB a cell plus 120 MB. Each slot's cache is its own
	// buffers (C6), so every slot can hold the full trained context.
	Slots int
	// PreemptChunk is the longest prefill chunk a background request runs,
	// so an interactive request that arrives mid-prefill waits one of these
	// rather than a whole -llm-batch (C2 rule 3). Zero takes Batch, i.e. no
	// cap. It applies only with more than one slot.
	PreemptChunk int
	// Reserve is how many slots only interactive requests may take, so
	// background work can never lock a voice command out. Ignored with one
	// slot.
	Reserve int
	// Class is the priority of a request that names none: "interactive" or
	// "background" (the default).
	Class string
	// NoCheckpoints turns off each slot's checkpoint at the last user turn
	// (C4), which is the control for measuring what it saves.
	NoCheckpoints bool
	// NoBatchDecode runs every decode step alone rather than batching the
	// steps of concurrent conversations into one pass (C5): the control.
	NoBatchDecode bool
}

const (
	defaultLLMModelID  = "qwen3.8-flash-next"
	defaultLLMContext  = 4096
	defaultLLMBatch    = 4096
	defaultLLMMaxToken = 1024
)

// LLM is the qwen3.8-flash-next adapter: an api.CompletionBackend over the
// graph cmd/llm drives.
//
// It is as resident as a model gets here. The whole checkpoint is staged at
// load -- 48 layers, the dense banks and the 512-expert bank, which is most
// of this machine's memory (LLM.md L6a) -- and a request costs a prefill and
// one forward pass per token, with no weight ever moving again.
//
// **The device lock is taken per forward pass and not per request.** A
// completion is seconds long and the speech models are milliseconds; holding
// the queue for a whole generation would make a transcription wait behind it
// for no gain, since nothing is in flight between two decode steps.
//
// **The graph belongs to the scheduler** (llm_sched.go, CONCURRENCY.md C2).
// It holds Slots conversations, a request holds one slot for its life, and
// the scheduler interleaves every live request's prefill chunks and decode
// steps on the device by priority and then by fair share of device time.
type LLM struct {
	opt   LLMOptions
	id    string
	model *llm.Model
	tok   *tokenizer.Tokenizer
	eog   map[int32]bool
	// The checkpoint's own recommended sampling (`general.sampling.*`),
	// which is what a request that names none gets.
	temp, topP float64
	topK       int

	class llmClass

	closeMu sync.Mutex
	g       *llm.Graph
	sched   *llmSched
}

// NewLLM loads the checkpoint and stages the whole model on the device.
//
// This is the expensive constructor in the repository: the weights are most
// of the machine's RAM and staging them is tens of seconds. It is done once,
// at startup, so that a request is a forward pass and nothing else.
func NewLLM(opt LLMOptions) (*LLM, error) {
	if opt.ID == "" {
		opt.ID = defaultLLMModelID
	}
	if opt.Context <= 0 {
		opt.Context = defaultLLMContext
	}
	if opt.Batch <= 0 {
		opt.Batch = defaultLLMBatch
	}
	if opt.MaxTokens <= 0 {
		opt.MaxTokens = defaultLLMMaxToken
	}
	if opt.Batch > opt.Context {
		opt.Batch = opt.Context
	}
	if opt.Slots <= 0 {
		opt.Slots = 1
	}
	if opt.PreemptChunk <= 0 || opt.PreemptChunk > opt.Batch || opt.Slots == 1 {
		opt.PreemptChunk = opt.Batch
	}
	if opt.Slots == 1 {
		opt.Reserve = 0
	}
	if opt.Reserve < 0 || opt.Reserve >= opt.Slots {
		return nil, fmt.Errorf("backend: %d of %d slots reserved for interactive requests leaves background none",
			opt.Reserve, opt.Slots)
	}
	class := classBackground
	if opt.Class != "" {
		c, ok := parseClass(opt.Class)
		if !ok {
			return nil, fmt.Errorf("backend: priority class %q; want interactive or background", opt.Class)
		}
		class = c
	}
	if opt.Device == nil {
		return nil, fmt.Errorf("backend: the language model has no host path; -gpu is required")
	}
	m, err := llm.Open(opt.Model)
	if err != nil {
		return nil, fmt.Errorf("backend: loading %s: %w", opt.Model, err)
	}
	tok, err := llm.LoadTokenizer(m.Set)
	if err != nil {
		_ = m.Close()
		return nil, fmt.Errorf("backend: %w", err)
	}
	l := &LLM{
		opt: opt, id: opt.ID, model: m, tok: tok, class: class,
		eog:  llm.EndOfGeneration(m.Set, tok),
		temp: metaFloat(m, "general.sampling.temp", 1),
		topP: metaFloat(m, "general.sampling.top_p", 0.95),
		topK: int(metaFloat(m, "general.sampling.top_k", 20)),
	}
	err = opt.Device.Do(func(dev *vk.Device) error {
		g, err := llm.NewGraph(dev, m, llm.GraphOpts{
			MaxTokens: opt.Batch, NKV: opt.Context, Layers: opt.Layers, Slots: opt.Slots,
		})
		if err != nil {
			return err
		}
		l.g = g
		return nil
	})
	if err != nil {
		_ = m.Close()
		return nil, fmt.Errorf("backend: staging %s: %w", opt.Model, err)
	}
	l.sched = newLLMSched(l.g, opt.Device, opt.Batch, opt.PreemptChunk, opt.Reserve, opt.NoBatchDecode)
	return l, nil
}

// metaFloat reads one of the checkpoint's recommended sampling settings,
// falling back to the model card's value when the key is absent.
func metaFloat(m *llm.Model, key string, def float64) float64 {
	if v, ok := m.Set.Float(key); ok {
		return v
	}
	return def
}

// Models reports the one model this backend serves.
func (l *LLM) Models() []api.Model {
	return []api.Model{{ID: l.id, Object: "model", OwnedBy: "local"}}
}

// Context is the cache's cell count, for the startup banner.
func (l *LLM) Context() int { return l.opt.Context }

// Slots is how many conversations it holds at once, for the same banner.
func (l *LLM) Slots() int { return l.opt.Slots }

// Close releases the device residency. Requests still running fail.
func (l *LLM) Close() {
	l.closeMu.Lock()
	defer l.closeMu.Unlock()
	if l.sched != nil {
		l.sched.close()
		l.sched = nil
	}
	if l.g != nil {
		_ = l.opt.Device.Do(func(*vk.Device) error {
			l.g.Destroy()
			return nil
		})
		l.g = nil
	}
	if l.model != nil {
		_ = l.model.Close()
		l.model = nil
	}
}

// Complete renders the conversation, prefills it and decodes until something
// stops it.
//
// The whole request is one sequence in one graph, which is why it is
// serialised: the attention cache, the PLE convolution's ring and every
// DeltaNet layer's recurrent state carry from token to token (LLM.md L7b),
// and they are the conversation. A second conversation interleaved into them
// would not be slower, it would be wrong.
func (l *LLM) Complete(ctx context.Context, req *api.CompletionRequest,
	emit func(api.Delta) error,
) (_ *api.CompletionResult, err error) {
	// enter is what the *client's* clock started at, as near as this side can
	// see it: everything below -- rendering the template, tokenizing, waiting
	// for the graph another request is holding, and the prefill -- is time
	// before the first token arrives, so it is all in the time to first token
	// this reports.
	enter := time.Now()
	msgs, opt, err := chatRequest(req)
	if err != nil {
		return nil, err
	}
	prompt, err := llm.RenderChat(msgs, opt)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	ids, err := l.tok.Encode(prompt)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", err, api.ErrUnsupported)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("the conversation tokenizes to nothing: %w", api.ErrUnsupported)
	}
	// The cache is the residency this process was started with, so a
	// conversation past it is refused with the number rather than truncated
	// at one end or the other -- both of which change the answer without
	// saying so.
	if len(ids) >= l.opt.Context {
		return nil, fmt.Errorf(
			"the conversation is %d tokens and this server staged a %d-cell cache; start it with a larger -llm-ctx: %w",
			len(ids), l.opt.Context, api.ErrUnsupported)
	}
	budget := req.Budget()
	if budget <= 0 {
		budget = l.opt.MaxTokens
	}
	if room := l.opt.Context - len(ids); budget > room {
		budget = room
	}

	class := l.classOf(ctx, req)
	l.closeMu.Lock()
	sched := l.sched
	l.closeMu.Unlock()
	if sched == nil {
		return nil, fmt.Errorf("the language model is closed")
	}
	job, err := sched.acquire(ctx, class, ids, l.checkpointAt(prompt, ids))
	if err != nil {
		return nil, err
	}
	defer sched.release(job)
	// queued is how long this request waited for a slot: every slot held by
	// another conversation. It is reported separately because it is the one
	// part of a slow time to first token that is not this request's own
	// work, and without it a turn that waited eight seconds looks like a
	// turn whose prefill was eight seconds. stalled is the other kind of
	// wait: time its own units spent queued behind other conversations'
	// units once it had a slot (CONCURRENCY.md C2).
	queued := time.Since(enter)
	reused, restored := job.reused, job.restore

	start := time.Now()
	logits, stalled, err := sched.run(job, ids[reused:], reused == 0)
	if err != nil {
		return nil, err
	}
	prefill := time.Since(start)

	s := l.sampler(req)
	dec := llm.NewChatDecoder(opt.Tools, !opt.NoThinking)
	dec.StopAt(req.Stop)
	var text utf8Stream
	res := &api.CompletionResult{FinishReason: "length"}
	gen := 0
	// ttft is taken at the first token the model produces rather than at the
	// first delta emitted: a byte-level token can be held back for a rune to
	// finish and a `<think>` marker is never emitted at all, so timing the
	// stream would attribute the decoder's own buffering to the model.
	var ttft, decode time.Duration
	start = time.Now()
	// A run that ends in an error is the one whose numbers are hardest to
	// come by afterwards -- a client that hangs up mid-stream leaves nothing
	// but "cancelled" in the access log -- so it reports the same line the
	// successful path does, up to where it stopped.
	defer func() {
		if err == nil || gen == 0 {
			return
		}
		if decode == 0 {
			decode = time.Since(start)
		}
		l.logRun(ctx, runStats{
			prompt: len(ids), reused: reused, restored: restored, gen: gen, class: class, slot: job.slot,
			queued: queued, stalled: stalled, prefill: prefill, ttft: ttft, decode: decode,
			total: time.Since(enter), reason: failureReason(ctx, err),
		})
	}()
	for gen < budget {
		id := s.Sample(logits)
		if ttft == 0 {
			ttft = time.Since(enter)
		}
		if l.eog[id] {
			res.FinishReason = "stop"
			break
		}
		piece, err := l.tok.Decode([]int32{id})
		if err != nil {
			return nil, err
		}
		// A token is bytes, not characters: byte-level BPE splits a
		// multi-byte rune across tokens routinely, and a delta cut there is
		// invalid UTF-8 that JSON would quietly replace with U+FFFD. What is
		// held back here is at most three bytes, and only until the next
		// token.
		if whole := text.write(piece); whole != "" {
			reasoning, content := dec.Write(whole)
			if reasoning != "" || content != "" {
				if err := emit(api.Delta{ReasoningContent: reasoning, Content: content}); err != nil {
					return nil, err
				}
			}
		}
		gen++
		if seq := dec.Stopped(); seq != "" {
			// A stop sequence ends the generation, and the sequence itself
			// is not part of the answer -- the decoder has already kept it
			// out of the stream.
			res.FinishReason, res.StopSequence = "stop", seq
			break
		}
		if gen == budget {
			break
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var wait time.Duration
		if logits, wait, err = sched.step(job, id); err != nil {
			return nil, err
		}
		stalled += wait
	}
	decode = time.Since(start)

	reasoning, content, calls, err := dec.Close()
	if err != nil {
		return nil, err
	}
	if reasoning != "" || content != "" {
		if err := emit(api.Delta{ReasoningContent: reasoning, Content: content}); err != nil {
			return nil, err
		}
	}
	for i, c := range calls {
		res.ToolCalls = append(res.ToolCalls, &api.ToolCall{
			ID: fmt.Sprintf("call_%08x", rand.Uint32()), Type: "function", Index: i,
			Function: api.ToolCallReq{Name: c.Name, Arguments: c.Arguments},
		})
	}
	if len(res.ToolCalls) > 0 {
		res.FinishReason = "tool_calls"
	}
	res.Usage = api.Usage{
		PromptTokens: len(ids), CompletionTokens: gen, TotalTokens: len(ids) + gen,
	}
	l.logRun(ctx, runStats{
		prompt: len(ids), reused: reused, restored: restored, gen: gen, class: class, slot: job.slot,
		queued: queued, stalled: stalled, prefill: prefill, ttft: ttft, decode: decode,
		total: time.Since(enter), reason: res.FinishReason,
	})
	return res, nil
}

// runStats is what one completion cost. The two rates are the two halves of
// what this server is: a prefill reads the whole prompt in one pass per batch
// and is bandwidth against the expert bank, a decode is one pass per token
// and is latency, and a single "tokens per second" over the pair would be
// neither number.
type runStats struct {
	reason              string
	prompt, reused, gen int
	// restored is whether the reused tokens came from the slot's checkpoint
	// (CONCURRENCY.md C4) rather than from what it held.
	restored bool
	class    llmClass
	slot     int
	// queued is the wait for a slot, prefill the prompt, ttft the whole
	// span from the call to the first token -- queue, render, tokenize and
	// prefill included -- decode the generation after it, and total the
	// wall clock a client saw. stalled is the part of prefill and decode its
	// units spent behind other conversations' units.
	queued, stalled, prefill, ttft, decode, total time.Duration
}

// logRun writes the one line a completion leaves in the journal.
//
// With more than one slot it also says which slot and class the request ran
// in and how long its units stalled behind other conversations', since a
// rate read off an interleaved run is the device's share and not its speed.
func (l *LLM) logRun(ctx context.Context, st runStats) {
	where := ""
	if l.opt.Slots > 1 {
		where = fmt.Sprintf(" [slot %d, %s%s]", st.slot, st.class, stalledNote(st.stalled))
	}
	log.Printf("%sllm %s%s: prompt %d tokens%s in %v (%.1f tok/s), ttft %s%s, generated %d tokens in %v (%.2f tok/s), %v total, %s",
		logID(ctx), l.id, where, st.prompt, reusedNote(st.reused, st.restored),
		st.prefill.Round(time.Millisecond), rate(st.prompt-st.reused, st.prefill),
		since(st.ttft), queuedNote(st.queued),
		st.gen, st.decode.Round(time.Millisecond), rate(st.gen, st.decode),
		st.total.Round(time.Millisecond), st.reason)
}

// stalledNote is the time a request's units waited behind other
// conversations', when there was any worth printing.
func stalledNote(d time.Duration) string {
	if d < 10*time.Millisecond {
		return ""
	}
	return fmt.Sprintf(", %v stalled", d.Round(time.Millisecond))
}

// classOf is a request's priority: the X-Priority header, then the body's
// service_tier, then the server's default (CONCURRENCY.md, "Priority").
func (l *LLM) classOf(ctx context.Context, req *api.CompletionRequest) llmClass {
	if c, ok := parseClass(strings.ToLower(api.Priority(ctx))); ok {
		return c
	}
	switch strings.ToLower(req.ServiceTier) {
	case "priority":
		return classInteractive
	case "flex":
		return classBackground
	}
	return l.class
}

// failureReason is what the log calls a run that did not finish. A client
// that hung up is the common one and is not an error in this server, so it is
// named rather than printed as one.
func failureReason(ctx context.Context, err error) string {
	if errors.Is(err, context.Canceled) || ctx.Err() != nil {
		return "cancelled"
	}
	return "failed: " + err.Error()
}

func rate(n int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}

// since prints a duration for the log, or "n/a" for a generation that never
// reached a first token -- a budget of nothing, or a stop sequence the prompt
// itself ended on. Zero is not a plausible measurement, so printing it as one
// would be a lie in the column a reader scans for the number.
func since(d time.Duration) string {
	if d <= 0 {
		return "n/a"
	}
	return d.Round(time.Millisecond).String()
}

// logID is the access log's request id, in the form every other line about a
// request carries it. A busy server interleaves these lines with the access
// log's own, and the id is what says which completion a rate belongs to.
func logID(ctx context.Context) string {
	if id := util.RequestID(ctx); id != "" {
		return "[" + id + "] "
	}
	return ""
}

// queuedNote accounts for time spent waiting on the graph, and only when
// there was some: the device lock is uncontended on a server answering one
// conversation, and a ", 0s queued" on every line would train the eye to skip
// the place where the number matters.
func queuedNote(d time.Duration) string {
	if d < 10*time.Millisecond {
		return ""
	}
	return fmt.Sprintf(" (%v queued)", d.Round(time.Millisecond))
}

// commonPrefix is how many leading tokens two sequences share.
func commonPrefix(a, b []int32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// reusedNote is the log's account of a continued conversation, which is the
// difference between a turn that costs its own history and one that does not.
// The prefill rate beside it is over the tokens actually run, so a turn that
// reused most of its prompt reports the speed of the part that cost anything
// rather than a figure flattered by the cache.
func reusedNote(reused int, restored bool) string {
	switch {
	case reused == 0:
		return ""
	case restored:
		return fmt.Sprintf(" (%d from a checkpoint)", reused)
	}
	return fmt.Sprintf(" (%d cached)", reused)
}

// minCheckpoint is the shortest prefix worth a checkpoint. Taking one copies
// every DeltaNet layer's state (~120 MB at 48 layers), so a prefix that
// prefills in less time than that copy takes is not worth it.
const minCheckpoint = 256

// checkpointAt is where this request's slot takes its checkpoint
// (CONCURRENCY.md C4): the end of everything before the last user turn, or 0
// for nowhere. That is the part the next request is likely to share, whether
// it is the next voice command on the same system prompt or the next turn of
// this conversation.
//
// It is found on the rendered text and then re-encoded rather than searched
// for in the ids. A special token splits the text before BPE runs, so the
// prefix's ids must be a prefix of the whole prompt's, and a re-encoding that
// is not one is refused rather than trusted.
func (l *LLM) checkpointAt(prompt string, ids []int32) int {
	if l.opt.NoCheckpoints {
		return 0
	}
	i := strings.LastIndex(prompt, "<|im_start|>user")
	if i <= 0 {
		return 0
	}
	pre, err := l.tok.Encode(prompt[:i])
	if err != nil || len(pre) < minCheckpoint || len(pre) >= len(ids) || commonPrefix(pre, ids) != len(pre) {
		return 0
	}
	return len(pre)
}

// sampler is the request's sampling, or the checkpoint's own where the
// request named none.
//
// A temperature of zero is greedy and is a legitimate request rather than an
// unset field, which is why the API's sampling fields are pointers. The seed
// is a clock unless the client named one: llm.NewSampler is deterministic at
// seed zero -- this is a reproducibility-first repository -- and a server
// that returned the same completion to every caller of the same prompt would
// be surprising in a way that is nobody's intent.
func (l *LLM) sampler(req *api.CompletionRequest) *llm.Sampler {
	temp, topP, topK := l.temp, l.topP, l.topK
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	if req.TopP != nil {
		topP = *req.TopP
	}
	if req.TopK != nil {
		topK = *req.TopK
	}
	seed := int64(rand.Uint64())
	if req.Seed != nil {
		seed = *req.Seed
	}
	return llm.NewSampler(float32(temp), topK, float32(topP), seed)
}

// chatRequest translates an HTTP request into the conversation the
// checkpoint's template reads. It is the one place the API's vocabulary and
// the model's meet.
func chatRequest(req *api.CompletionRequest) ([]llm.ChatMessage, llm.ChatOpts, error) {
	var opt llm.ChatOpts
	switch strings.ToLower(req.ReasoningEffort) {
	case "":
	case "none", "minimal":
		// OpenAI's two spellings for "do not think". The template has one
		// too, and it enforces it in the prompt rather than asking.
		opt.NoThinking = true
	default:
		opt.Effort = strings.ToLower(req.ReasoningEffort)
	}

	choice, err := toolChoice(req.ToolChoice)
	if err != nil {
		return nil, opt, err
	}
	if choice != "none" {
		for i := range req.Tools {
			// The bytes the client sent, not this struct re-encoded: see
			// api.Tool.Raw.
			opt.Tools = append(opt.Tools, llm.ChatTool{JSON: req.Tools[i].Raw()})
		}
	}

	msgs := make([]llm.ChatMessage, 0, len(req.Messages))
	for i, m := range req.Messages {
		role := m.Role
		if role == "" {
			role = "user"
		}
		out := llm.ChatMessage{
			Role:      role,
			Content:   m.Content.Text(),
			Reasoning: m.ReasoningContent,
		}
		for _, call := range m.ToolCalls {
			if call == nil {
				continue
			}
			args, err := llm.ParseToolArguments(call.Function.Arguments)
			if err != nil {
				return nil, opt, fmt.Errorf(
					"message %d: tool call %q has arguments that are not a JSON object: %v: %w",
					i, call.Function.Name, err, api.ErrUnsupported)
			}
			out.ToolCalls = append(out.ToolCalls, llm.ChatToolCall{Name: call.Function.Name, Args: args})
		}
		msgs = append(msgs, out)
	}
	return msgs, opt, nil
}

// toolChoice reads OpenAI's `tool_choice`, which is a string or an object.
// "auto" and "none" are the two this server can honour -- the first is what
// the tool block already asks for and the second is leaving it out -- and
// forcing a particular call is constrained decoding, which does not exist
// here yet.
func toolChoice(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "auto", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", fmt.Errorf(
			"tool_choice naming a function is not implemented; this server has no constrained decoding: %w",
			api.ErrUnsupported)
	}
	switch s {
	case "auto", "none":
		return s, nil
	default:
		return "", fmt.Errorf(
			"tool_choice %q is not implemented; this server takes \"auto\" and \"none\": %w",
			s, api.ErrUnsupported)
	}
}

// utf8Stream holds back the trailing bytes of a partial rune.
//
// The model's tokens are bytes: byte-level BPE splits a multi-byte character
// across two tokens routinely, and "🍵" arrives as four tokens of one byte
// each. Emitting those as they come would put four replacement characters in
// the stream and one in the buffered answer.
type utf8Stream struct{ tail []byte }

// write returns the part of the stream that is now whole runes, keeping at
// most three bytes back.
func (u *utf8Stream) write(piece string) string {
	u.tail = append(u.tail, piece...)
	cut := len(u.tail)
	// A rune is at most four bytes, so the last rune is the only one that
	// can be partial: find where it starts and ask whether it is finished.
	for i := 1; i <= 4 && i <= len(u.tail); i++ {
		start := len(u.tail) - i
		if u.tail[start]&0xC0 == 0x80 {
			continue // a continuation byte; keep walking back
		}
		if !utf8.FullRune(u.tail[start:]) {
			cut = start
		}
		break
	}
	out := string(u.tail[:cut])
	u.tail = append(u.tail[:0], u.tail[cut:]...)
	return out
}
