package bench

// The matrix shapes the models in GOALS.md are actually built from.
//
// Every other family in this suite sweeps square power-of-two N x N x N,
// which no transformer layer is. IDEAS §3.4 asked for the real triples
// instead, on the argument that they refocus everything else: the load-width
// rule (§1.7) is per matrix, the tile choice is per shape, and a kernel that
// wins at 4096x4096x4096 has not been shown to win at 2048x640x2560.
//
// Provenance, so these can be re-derived rather than trusted:
//
//   - z-image-turbo and its text encoder are read from the local checkout,
//     models/Z-Image-Turbo/{transformer,text_encoder}/config.json, with the
//     feed-forward width taken from the safetensors headers themselves
//     (layers.0.feed_forward.w1.weight is [10240, 3840]) because the
//     transformer config does not state it.
//   - qwen3.8-flash-next is huggingface.co/Qwen/Qwen3.8-Flash-Next
//     config.json "text_config" (48 layers, 12 of them full attention on a
//     full_attention_interval of 4, 512-expert MoE with 10 active).
//   - parakeet-tdt-0.6b-v3 is nvidia/parakeet-tdt-0.6b-v3 config.json
//     (24-layer conformer encoder, d=1024, ffn=4096, 8x subsampling).
//   - kokoro-82m is hexgrad/Kokoro-82M config.json (the PL-BERT block is
//     the only attention stack; the rest is convolutional or recurrent).
//   - qwen3-embedding-0.6b is Qwen/Qwen3-Embedding-0.6B config.json.
//
// A weight matrix is written [N, K] — out features by in features — and the
// activation batch is M, so every row here is C[M,N] = A[M,K] * B[K,N], the
// same convention the GEMM kernels take. Decode rows are M=1 and are GEMV.
type modelShape struct {
	model string
	layer string
	M     int // tokens in the batch: 1 for decode, the real batch otherwise
	N     int // out features
	K     int // in features, i.e. the reduction length
	// count is how many times this exact matmul runs in one forward pass at
	// this M, so that a pass's total weight traffic can be added up rather
	// than eyeballed. For the MoE rows it already accounts for how many
	// experts are touched — ten per token at decode, all 512 at prefill.
	count int
	note  string
}

// Why each M was chosen, since the token count is the one dimension no
// config file states:
//
//	qwen3.8-flash-next  prefill 2048 is one prompt chunk. The expert rows run
//	                    at 2048*10/512 = 40, the mean tokens routed to one of
//	                    512 experts at top-10 — the shape nothing in the
//	                    square sweep resembles.
//	parakeet            384 frames is ~30 s of audio: 100 frames/s at 10 ms
//	                    hop, subsampling_factor 8, so 12.5 frames/s.
//	kokoro              128 phonemes is a sentence; 512 is the PL-BERT
//	                    max_position_embeddings, i.e. the longest it takes.
//	z-image-turbo       4096 latent tokens is a 1024x1024 image: 128x128
//	                    latents at the VAE's 8x, then patch size 2 -> 64x64.
//	                    1024 is the same for 512x512. The text encoder runs
//	                    at the caption length, 128.
//	qwen3-embedding     512 and 2048 are typical document chunks.
var modelShapes = []modelShape{
	// ---- qwen3.8-flash-next, decode (the tok/s path) ----------------------
	// 12 full-attention layers: 24 query heads and 2 KV heads of head_dim
	// 256, so the query projection is 8x the width of either KV projection.
	{model: "qwen3.8-flash-next", layer: "attn.q", M: 1, N: 6144, K: 2560, count: 12},
	{model: "qwen3.8-flash-next", layer: "attn.kv", M: 1, N: 512, K: 2560, count: 24, note: "2 KV heads x 256"},
	{model: "qwen3.8-flash-next", layer: "attn.o", M: 1, N: 2560, K: 6144, count: 12},
	// 36 gated-DeltaNet layers: q and k are 16 heads x 128, v and its output
	// gate are 48 heads x 128.
	{model: "qwen3.8-flash-next", layer: "la.qk", M: 1, N: 2048, K: 2560, count: 72},
	{model: "qwen3.8-flash-next", layer: "la.vz", M: 1, N: 6144, K: 2560, count: 72},
	{model: "qwen3.8-flash-next", layer: "la.o", M: 1, N: 2560, K: 6144, count: 36},
	// MoE, every layer: a 512-way router, ten active experts of width 640,
	// and one shared expert always on.
	{model: "qwen3.8-flash-next", layer: "moe.router", M: 1, N: 512, K: 2560, count: 48},
	{model: "qwen3.8-flash-next", layer: "moe.gate_up", M: 1, N: 640, K: 2560, count: 960, note: "10 of 512 experts x gate+up"},
	{model: "qwen3.8-flash-next", layer: "moe.down", M: 1, N: 2560, K: 640, count: 480, note: "10 of 512 experts"},
	{model: "qwen3.8-flash-next", layer: "moe.shared", M: 1, N: 640, K: 2560, count: 96},
	{model: "qwen3.8-flash-next", layer: "moe.shared_down", M: 1, N: 2560, K: 640, count: 48},
	// One matrix, 636M weights: 248320 vocab from 2560 hidden.
	{model: "qwen3.8-flash-next", layer: "lm_head", M: 1, N: 248320, K: 2560, count: 1, note: "vocab 248320"},

	// ---- qwen3.8-flash-next, prefill at 2048 tokens ----------------------
	{model: "qwen3.8-flash-next", layer: "attn.q", M: 2048, N: 6144, K: 2560, count: 12},
	{model: "qwen3.8-flash-next", layer: "attn.kv", M: 2048, N: 512, K: 2560, count: 24},
	{model: "qwen3.8-flash-next", layer: "attn.o", M: 2048, N: 2560, K: 6144, count: 12},
	{model: "qwen3.8-flash-next", layer: "la.qk", M: 2048, N: 2048, K: 2560, count: 72},
	{model: "qwen3.8-flash-next", layer: "la.vz", M: 2048, N: 6144, K: 2560, count: 72},
	{model: "qwen3.8-flash-next", layer: "la.o", M: 2048, N: 2560, K: 6144, count: 36},
	{model: "qwen3.8-flash-next", layer: "moe.router", M: 2048, N: 512, K: 2560, count: 48},
	// The shape the square sweep has no analogue for: 2048 tokens spread
	// over 512 experts is 40 rows each, and every expert in the bank is
	// touched, so prefill reads the whole 120B-parameter model rather than
	// the ~5B a decode step reads.
	{model: "qwen3.8-flash-next", layer: "moe.gate_up", M: 40, N: 640, K: 2560, count: 49152, note: "2048 tok/512 experts; all 512 touched"},
	{model: "qwen3.8-flash-next", layer: "moe.down", M: 40, N: 2560, K: 640, count: 24576, note: "all 512 experts touched"},
	{model: "qwen3.8-flash-next", layer: "moe.shared", M: 2048, N: 640, K: 2560, count: 96},
	{model: "qwen3.8-flash-next", layer: "moe.shared_down", M: 2048, N: 2560, K: 640, count: 48},

	// ---- parakeet-tdt-0.6b-v3: conformer encoder over ~30 s of audio ------
	{model: "parakeet-tdt-0.6b", layer: "enc.qkv", M: 384, N: 1024, K: 1024, count: 72},
	{model: "parakeet-tdt-0.6b", layer: "enc.o", M: 384, N: 1024, K: 1024, count: 24},
	{model: "parakeet-tdt-0.6b", layer: "enc.ff.up", M: 384, N: 4096, K: 1024, count: 48, note: "two half-FFN modules per layer"},
	{model: "parakeet-tdt-0.6b", layer: "enc.ff.down", M: 384, N: 1024, K: 4096, count: 48},
	{model: "parakeet-tdt-0.6b", layer: "enc.conv.pw1", M: 384, N: 2048, K: 1024, count: 24, note: "gated pointwise"},
	{model: "parakeet-tdt-0.6b", layer: "enc.conv.pw2", M: 384, N: 1024, K: 1024, count: 24},
	// The same encoder at ~82 s, to see how the shapes scale with M alone.
	{model: "parakeet-tdt-0.6b", layer: "enc.ff.up", M: 1024, N: 4096, K: 1024, count: 48},
	{model: "parakeet-tdt-0.6b", layer: "enc.ff.down", M: 1024, N: 1024, K: 4096, count: 48},
	// The TDT decoder runs one step per emitted token: a 2-layer LSTM of
	// hidden 640 (four gates, so 2560 out) and a joint network whose output
	// is the 8193-token vocabulary plus the five duration classes.
	{model: "parakeet-tdt-0.6b", layer: "dec.lstm", M: 1, N: 2560, K: 640, count: 4, note: "4 gates x 640; ih and hh"},
	{model: "parakeet-tdt-0.6b", layer: "joint.enc", M: 1, N: 640, K: 1024, count: 1},
	{model: "parakeet-tdt-0.6b", layer: "joint.out", M: 1, N: 8198, K: 640, count: 1, note: "8193 vocab + 5 durations"},

	// ---- kokoro-82m: the PL-BERT stack is the only attention block -------
	{model: "kokoro-82m", layer: "plbert.qkv", M: 128, N: 768, K: 768, count: 36},
	{model: "kokoro-82m", layer: "plbert.o", M: 128, N: 768, K: 768, count: 12},
	{model: "kokoro-82m", layer: "plbert.ff.up", M: 128, N: 2048, K: 768, count: 12},
	{model: "kokoro-82m", layer: "plbert.ff.down", M: 128, N: 768, K: 2048, count: 12},
	{model: "kokoro-82m", layer: "plbert.ff.up", M: 512, N: 2048, K: 768, count: 12, note: "max_position_embeddings"},
	// The prosody/duration path is LSTM at hidden_dim 512: four gates of 512.
	{model: "kokoro-82m", layer: "lstm.gates", M: 128, N: 2048, K: 512, count: 6},

	// ---- z-image-turbo: 30 DiT layers of width 3840, FFN 10240 -----------
	{model: "z-image-turbo", layer: "dit.qkv", M: 4096, N: 3840, K: 3840, count: 90},
	{model: "z-image-turbo", layer: "dit.o", M: 4096, N: 3840, K: 3840, count: 30},
	{model: "z-image-turbo", layer: "dit.ff.w13", M: 4096, N: 10240, K: 3840, count: 60},
	{model: "z-image-turbo", layer: "dit.ff.w2", M: 4096, N: 3840, K: 10240, count: 30},
	// The same model generating 512x512 instead of 1024x1024.
	{model: "z-image-turbo", layer: "dit.qkv", M: 1024, N: 3840, K: 3840, count: 90},
	{model: "z-image-turbo", layer: "dit.ff.w13", M: 1024, N: 10240, K: 3840, count: 60},
	{model: "z-image-turbo", layer: "dit.ff.w2", M: 1024, N: 3840, K: 10240, count: 30},
	{model: "z-image-turbo", layer: "cap_embedder", M: 128, N: 3840, K: 2560, count: 1},

	// ---- z-image-turbo's text encoder: Qwen3, 36 layers, over a caption --
	{model: "z-image-te", layer: "te.q", M: 128, N: 4096, K: 2560, count: 36},
	{model: "z-image-te", layer: "te.kv", M: 128, N: 1024, K: 2560, count: 72},
	{model: "z-image-te", layer: "te.o", M: 128, N: 2560, K: 4096, count: 36},
	{model: "z-image-te", layer: "te.ff.gate_up", M: 128, N: 9728, K: 2560, count: 72},
	{model: "z-image-te", layer: "te.ff.down", M: 128, N: 2560, K: 9728, count: 36},

	// ---- qwen3-embedding-0.6b: 28 layers, d=1024, FFN 3072 ---------------
	{model: "qwen3-embedding", layer: "emb.q", M: 512, N: 2048, K: 1024, count: 28},
	{model: "qwen3-embedding", layer: "emb.kv", M: 512, N: 1024, K: 1024, count: 56},
	{model: "qwen3-embedding", layer: "emb.o", M: 512, N: 1024, K: 2048, count: 28},
	{model: "qwen3-embedding", layer: "emb.ff.gate_up", M: 512, N: 3072, K: 1024, count: 56},
	{model: "qwen3-embedding", layer: "emb.ff.down", M: 512, N: 1024, K: 3072, count: 28},
	{model: "qwen3-embedding", layer: "emb.ff.gate_up", M: 2048, N: 3072, K: 1024, count: 56},
	{model: "qwen3-embedding", layer: "emb.ff.down", M: 2048, N: 1024, K: 3072, count: 28},
}
