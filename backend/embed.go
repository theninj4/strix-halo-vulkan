package backend

import (
	"context"
	"fmt"
	"sync"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/embed"
	"strix-halo-vulkan/vk"
)

// EmbedOptions is what cmd/serve's flags come to.
type EmbedOptions struct {
	// Model is the Qwen3-Embedding-0.6B checkpoint directory.
	Model string
	// Device runs the model on the GPU. Without it this is the fp32 CPU
	// reference, which is two seconds a text rather than twelve
	// milliseconds -- correct, and not a thing to serve.
	Device *Device
	// MaxTokens is the longest run the arenas are sized for, and therefore
	// where an input is truncated. Zero takes defaultEmbedTokens.
	MaxTokens int
	// ID is the model id this backend answers to in /v1/models. Empty takes
	// defaultEmbedModelID.
	ID string
}

const (
	defaultEmbedModelID = "qwen3-embedding-0.6b"
	// defaultEmbedTokens is the arena size, not the model's limit: the
	// checkpoint handles 32k positions, and the arenas are what a run costs
	// up front. 512 covers a query and most documents; a corpus of long
	// documents wants -embed-tokens raised and the memory that comes with
	// it.
	defaultEmbedTokens = 512
)

// Embed is the Qwen3-Embedding adapter: an api.EmbeddingBackend over the
// model cmd/embed drives.
//
// One mutex guards the run. The device path is not reentrant -- one set of
// arenas, one residual stream -- and on the GPU every request would serialise
// at the queue anyway (see Device). The CPU path is guarded by the same lock
// for the simpler reason that it is only there to be correct.
type Embed struct {
	opt EmbedOptions
	id  string

	mu  sync.Mutex
	gpu *embed.GPU
	cpu *embed.Model
}

// NewEmbed loads the checkpoint. On the device that is 0.88 GB of fp16 banks
// staged in about 1.3 s; on the host it is 2.4 GB of fp32 and about 2 s.
func NewEmbed(opt EmbedOptions) (*Embed, error) {
	if opt.ID == "" {
		opt.ID = defaultEmbedModelID
	}
	if opt.MaxTokens <= 0 {
		opt.MaxTokens = defaultEmbedTokens
	}
	b := &Embed{opt: opt, id: opt.ID}
	if opt.Device == nil {
		m, err := embed.Load(opt.Model)
		if err != nil {
			return nil, err
		}
		b.cpu = m
		return b, nil
	}
	err := opt.Device.Do(func(dev *vk.Device) error {
		g, err := embed.NewGPU(dev, opt.Model, opt.MaxTokens)
		if err != nil {
			return err
		}
		b.gpu = g
		return nil
	})
	if err != nil {
		return nil, err
	}
	return b, nil
}

// Close releases the device objects.
func (b *Embed) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.gpu != nil {
		_ = b.opt.Device.Do(func(*vk.Device) error { b.gpu.Destroy(); return nil })
		b.gpu = nil
	}
	b.cpu = nil
}

// Models is what GET /v1/models reports.
func (b *Embed) Models() []api.Model {
	return []api.Model{{ID: b.id, Object: "model", OwnedBy: "qwen"}}
}

// MaxTokens is where an input is truncated.
func (b *Embed) MaxTokens() int { return b.opt.MaxTokens }

// Embed runs every input in the request, in order.
//
// The inputs are run one at a time. That is not an oversight: the encoder's
// graph is one causal sequence, so two texts in one pass would need a
// block-diagonal mask the attention kernel does not have (EMBEDDING.md E7),
// and concatenating them without one would let the second text attend to the
// first -- a wrong answer rather than a slow one.
func (b *Embed) Embed(ctx context.Context, req *api.EmbeddingRequest) (*api.EmbeddingResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.gpu == nil && b.cpu == nil {
		return nil, fmt.Errorf("backend: the embedding model is closed")
	}
	if req.Dimensions > embed.Dim {
		return nil, fmt.Errorf("%w: dimensions is %d; this model produces %d",
			api.ErrUnsupported, req.Dimensions, embed.Dim)
	}

	res := &api.EmbeddingResult{Vectors: make([][]float32, len(req.Input))}
	for i, text := range req.Input {
		// The instruction is prepended here rather than by the caller
		// because it is part of the *text* the model reads, so it has to be
		// inside the token count the response reports.
		if req.Instruct != "" {
			task := req.Instruct
			if task == "default" {
				task = embed.DefaultTask
			}
			text = embed.Instruct(task, text)
		}
		// Cancellation is checked between inputs. Inside one there is
		// nothing to check: a run is a submitted command buffer, and coming
		// back from it early would leave the device mid-graph.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ids, err := b.encode(text)
		if err != nil {
			return nil, err
		}
		var v []float32
		if b.gpu != nil {
			err = b.opt.Device.Do(func(*vk.Device) error {
				var e error
				v, e = b.gpu.EmbedIDs(ids)
				return e
			})
		} else {
			v, err = b.cpu.EmbedIDs(ids)
		}
		if err != nil {
			return nil, err
		}
		res.Vectors[i] = embed.Truncate(v, req.Dimensions)
		res.Usage.PromptTokens += len(ids)
	}
	res.Usage.TotalTokens = res.Usage.PromptTokens
	return res, nil
}

// encode tokenizes and truncates. Truncation keeps the *front* of the text
// and re-appends the end-of-text token, which is what HF's
// `truncation=True` does and what last-token pooling requires: the pooled row
// is the appended token's, so a truncation that dropped it would pool a
// different token than every other input did.
func (b *Embed) encode(text string) ([]int32, error) {
	tok := b.cpuTokenizer()
	return tok.EncodeLimit(text, b.opt.MaxTokens)
}

func (b *Embed) cpuTokenizer() *embed.Tokenizer {
	if b.gpu != nil {
		return b.gpu.Tok
	}
	return b.cpu.Tok
}
