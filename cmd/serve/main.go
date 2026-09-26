// Command serve is the long-lived HTTP server: the models in this repository
// behind an OpenAI-shaped API.
//
// What it loads is what it is told to load. Each vertical is a flag, and a
// vertical that was not asked for is not in memory and answers 501 with a
// message naming the flag that would have loaded it:
//
//	go run ./cmd/serve -tts -stt
//	go run ./cmd/serve -llm                         # the language model, 84 GB resident
//	go run ./cmd/serve -llm -llm-layers 4           # the server, in 7 GB, for the HTTP side
//	go run ./cmd/serve -tts -voice bm_george -addr :8080
//	go run ./cmd/serve -stt -max-audio 300          # five-minute clips
//	go run ./cmd/serve -embed                       # embeddings, 0.88 GB resident
//	go run ./cmd/serve -kev                         # System One classification (Kev-4B), ~9 GB resident
//	go run ./cmd/serve -image                       # qwen-image-2.1, 32 GB resident
//	go run ./cmd/serve -image -image-size 512x512   # a quarter of the tokens, a quarter of the arenas
//	go run ./cmd/serve -video                       # minimax-h3 video jobs; ~0.3 GB at rest, ~50 GB a request
//	go run ./cmd/serve -tts -tts-gpu=false          # the CPU reference
//	go run ./cmd/serve -tts -stt -wyoming :10300    # and the same two over Wyoming
//
// Every endpoint answers today. POST /v1/images/edits needs `-edits N`, where
// N is how many reference images an edit may carry: an edit in
// Qwen-Image-2.1 is a conditional generation over a vision tower rather than
// an SDEdit, so the references are rows of the transformer's own sequence and
// their cost is residency (IMAGE.md Q8).
//
// **There is no -preview flag any more, and that is the change rather than
// an omission.** Under Z-Image it loaded madebyollin/taef1 and cost 1.0 GB
// of activation arena, so streaming was a residency decision. This model has
// no distilled decoder to load: its preview is a fitted 64x4 matrix
// (IMAGE.md Q7), 260 float32s compiled in, so `stream: true` is always
// answerable and there is nothing to turn on.
//
// **-wyoming is a second door onto the speech backends**, not a second copy
// of them: one process, one staging, one GPU queue, answering Home
// Assistant's protocol on its own port beside the HTTP API on -addr. It takes
// an address rather than a boolean because the address is the decision that
// matters -- Wyoming has no notion of a credential, so -token does not reach
// it and anything that can open the port can run the two speech models on
// it. See the `wyoming` package.
//
// **-image-size is a ceiling, it is an *area*, and it has a hard limit.** The
// arenas are allocated once for it and every request inside that many pixels
// runs in them, so what the number buys is the largest image and nothing
// else. The shape is free: every arena in the image pipeline is sized by the
// pixel count alone (`qimage/vae`'s TestArenaShape measures the VAE's at 3060
// bytes a pixel for every aspect ratio), so `-image-size 1024x1024` answers
// `aspect_ratio: "16:9"` with 1344x768 rather than with 1024x576. The limit
// is 1,403,584 pixels -- 1184x1184 square, 1536x864 at 16:9 -- and it is not
// a budget: the VAE decoder's activation arena is a single storage buffer and
// this device caps one at 4 GiB - 4, which a 1216x1216 decode is already
// past. Anything larger is refused at startup with the arithmetic.
//
// **-llm stages D19's widths and D20's MoE bank by default** (P3a, P4b).
// D19 is D18's 4.5-bit plan with `ple_proj` on int8, plus P3a's fifth bit on
// `full_attn`, `qsa_indexer`, `lm_head` and `hyper_conn`. D20 narrows the
// four MoE rows that ship at 8.5 bits and have a format the kernels already
// read -- the shared expert's gate and up to Q4_K, its down and the five
// Q8_0 layers of `ffn_down_exps` to Q5_1. Together: **4.279 GB a token,
// 4.0970 perplexity (+1.69% against our own 4.0289) and 35.52 tok/s,
// 1.41x llama.cpp**. `LLM_DENSE_BANK` and `LLM_MOE_BANK` override by naming
// any other plan, and `off` serves the wider bank instead. `cmd/llm` has no
// such default on purpose -- a measurement tool should stage only what its
// command line names.
//
// **-llm and -image do not fit together.** The language model is ~84 GB
// resident and the image pipeline ~25 GB, against 128 GB of unified memory
// that the rest of the machine is also in; the two are separate processes on
// this part, or separate runs.
//
// Two things about the device are worth knowing before reading the timings.
// **Requests are serialised at the queue**, because a vk.Device has one and
// nothing in `vk` is externally synchronised -- the server is concurrent at
// the HTTP layer and serial at the GPU, which is also the only way these
// models make sense on a part with one 236 GB/s bus. And **-tts-gpu stages
// once**: kokoro's arenas are sized for a ceiling -- -tts-frames, 25 s of
// speech by default -- and every shorter utterance is a prefix of them, so a
// request costs the model and not the staging. It was the other way round
// until T9, which is why the flag used to default off; see
// backend.TTS.synthesize.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/backend"
	"strix-halo-vulkan/llm"
	"strix-halo-vulkan/wyoming"
)

// defaultToken is what the API has always used. It is a flag so that a
// deployment can change it, and an empty one opens the server, which is only
// sensible on loopback.
const defaultToken = "womblesofwimbledon"

// defaultLLMModel names the shard rather than the directory it is in, because
// that directory now holds three checkpoints -- the model, Unsloth's imatrix
// and the MTP draft head -- and a directory holding more than one is
// ambiguous rather than convenient.
const defaultLLMModel = "models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

func main() {
	log.SetFlags(log.Ltime)
	addr := flag.String("addr", "127.0.0.1:8080", "address to listen on; loopback by default")
	token := flag.String("token", defaultToken, "bearer token; empty serves without authorization")
	gpu := flag.Bool("gpu", true, "run the models on Vulkan where they have a resident path")
	maxUpload := flag.Int64("max-upload", 128, "largest request body that carries a file, in MB")
	logBodies := flag.Bool("log-bodies", false,
		"print each request and response body in the access log; off because they are conversations")

	tts := flag.Bool("tts", false, "load Kokoro-82M and serve /v1/audio/speech")
	ttsModel := flag.String("tts-model", "models/Kokoro-82M", "converted kokoro checkpoint directory")
	ttsGPU := flag.Bool("tts-gpu", true, "run kokoro on the device")
	ttsFrames := flag.Int("tts-frames", 0,
		"longest utterance the kokoro arenas are staged for, in 25 ms frames; 0 takes the backend's default")
	lexicon := flag.String("lexicon", "models/misaki", "misaki lexicon directory; empty takes phonemes only")
	espeak := flag.Bool("espeak", true, "use espeak-ng for words outside the lexicon")
	british := flag.Bool("british", false, "use the en-GB lexicon and fallback")
	voice := flag.String("voice", "af_heart", "voice a request that names none gets; a comma-joined mix is one too")
	noise := flag.Int64("noise", 0, "seed for the vocoder's excitation noise; 0 leaves it off")

	llmOn := flag.Bool("llm", false, "load qwen3.8-flash-next and serve /v1/chat/completions")
	llmModel := flag.String("llm-model", defaultLLMModel, "GGUF checkpoint, shard or directory")
	llmCtx := flag.Int("llm-ctx", 4096, "cache cells: the longest conversation, prompt plus completion. "+
		"Up to 262144, the model's trained context (P18): 7.3 GB of KV cache, ~83 GB resident with -llm-batch 8192")
	llmBatch := flag.Int("llm-batch", 4096,
		"tokens the prefill arenas hold; a longer prompt is prefilled in chunks of it. "+
			"P16, through this server: 2048 is 1052 tok/s prefill, 4096 is 1199 (1.14x), 8192 is 1233, and decode is 34.2 at all three. "+
			"Wider costs only memory: 8192 helps prompts past 4096 tokens and holds ~3 GB more")
	llmMax := flag.Int("llm-max-tokens", 1024, "tokens a request that names no max_tokens gets")
	llmLayers := flag.Int("llm-layers", 0, "stage only the first N layers; 0 is the model, anything else is a fast start and not an answer")
	llmSlots := flag.Int("llm-slots", 1, "conversations held at once, interleaved a prefill chunk or decode step at a time (CONCURRENCY.md). "+
		"Each is its own cache of -llm-ctx cells, ~27.8 KB a cell")
	llmPreempt := flag.Int("llm-preempt-chunk", 2048, "with more than one slot, the longest prefill chunk a background request runs "+
		"before an interactive one may take the device; 0 is -llm-batch")
	llmBatchDecode := flag.Bool("llm-batch-decode", true, "with more than one slot, advance the decoding conversations one token each "+
		"in one pass rather than a pass each (CONCURRENCY.md C5)")
	llmCheckpoints := flag.Bool("llm-checkpoints", true, "keep each slot's state at the end of everything before the last user turn, "+
		"so the next request sharing that prefix (a voice command on the same system prompt) prefills only the rest (CONCURRENCY.md C4)")
	llmReserve := flag.Int("llm-reserve", 1, "with more than one slot, how many only interactive requests may take")
	llmClass := flag.String("llm-class", "background", "priority of a request that names none (X-Priority header, "+
		"or service_tier \"priority\"/\"flex\"): interactive or background")
	llmPresets := flag.String("llm-presets", "models.ini", "llama-server preset file: each section is a model name a "+
		"chat request can ask for, with its own sampling and thinking defaults over the same weights. "+
		"The default is skipped if it does not exist; empty turns presets off")
	llmMMProj := flag.String("llm-mmproj", "", "the vision tower's mmproj GGUF, e.g. "+
		"models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf; empty serves text only and refuses images (LLM-VISION.md)")
	llmVisionTokens := flag.Int("llm-vision-tokens", 4096, "the most tokens one image becomes (32x32 pixels a token, "+
		"so 4096 is a 2048x2048 picture); capped at -llm-batch")
	llmVisionImages := flag.Int("llm-vision-images", 8, "the most images one request may carry")
	llmProgress := flag.Duration("llm-progress", 10*time.Second, "how often to log each in-flight completion's "+
		"prefill/generation progress and rates; 0 turns it off")

	embedOn := flag.Bool("embed", false, "load Qwen3-Embedding-0.6B and serve /v1/embeddings")
	embedModel := flag.String("embed-model", "models/Qwen3-Embedding-0.6B", "embedding checkpoint directory")
	embedTokens := flag.Int("embed-tokens", 512, "longest input the embedding arenas hold; longer inputs are truncated")

	kevOn := flag.Bool("kev", false, "load Kev-4B and serve POST /v1/systemone (TypeSafe's System One, CLASSIFICATION.md)")
	kevModel := flag.String("kev-model", "models/kev-4b", "Kev checkpoint directory: adapter, converted head, tokenizer")
	kevBase := flag.String("kev-base", "models/Qwen3.5-4B-Base", "the Qwen3.5 base the Kev checkpoint was trained on")
	kevTokens := flag.Int("kev-tokens", 8192, "the longest request one pass holds, in packed tokens (the state once plus every question)")
	kevCache := flag.Int("kev-cache", 4, "states the prefix cache keeps, so a repeated text pays for its questions only; 0 turns it off")
	kevCacheTokens := flag.Int("kev-cache-tokens", 4096, "the longest state the prefix cache keeps (32 KB of KV a token, plus 52.7 MB a state)")
	kevBatch := flag.Int("kev-batch", 8, "the most requests one pass answers: a burst shares passes, a lone request waits for nothing (K7.5)")
	kevFP16 := flag.Bool("kev-fp16", false, "stage Kev's weights as fp16 instead of int8: the control, 1.27x slower and 3.3 GB more")

	stt := flag.Bool("stt", false, "load parakeet-tdt-0.6b-v3 and serve /v1/audio/transcriptions")
	sttModel := flag.String("stt-model", "models/parakeet-tdt-0.6b-v3", "parakeet checkpoint directory")
	maxAudio := flag.Float64("max-audio", 60, "longest clip the transcription arenas are sized for, in seconds")

	wyAddr := flag.String("wyoming", "",
		"also serve the Wyoming protocol (Home Assistant) on this address, e.g. :10300; empty is off")
	wyName := flag.String("wyoming-name", wyoming.DefaultName,
		"program name the Wyoming describe/info message advertises, "+
			"which is what Home Assistant names its entities")
	wyAllVoices := flag.Bool("wyoming-all-voices", false,
		"advertise every kokoro voice pack over Wyoming, not just the ones -lexicon can pronounce")
	wyRate := flag.Int("wyoming-rate", 0,
		"resample synthesised speech to this rate before sending it; 0 sends the model's own 24 kHz")
	wyMaxConns := flag.Int("wyoming-max-conns", 0,
		"concurrent Wyoming connections; 0 takes the package default")

	imgOn := flag.Bool("image", false, "load Qwen-Image-2.1 and serve /v1/images/generations")
	imgModel := flag.String("image-model", "models/Qwen-Image-2.1", "Qwen-Image-2.1 checkpoint root")
	imgSize := flag.String("image-size", "1024x1024",
		"largest image the arenas are built for, and the size a request that names none gets; its "+
			"*area* is the ceiling and any shape of that area is served, both sides a multiple of 32, "+
			"and no more than 1403584 pixels / 1184x1184 (the VAE's single-buffer arena)")
	imgSteps := flag.Int("image-steps", 40, "denoising steps a request that names none gets; the checkpoint's default is 40")
	imgPrompt := flag.Int("image-max-prompt", 512, "longest prompt the image text encoder is built for, in tokens")
	imgEdits := flag.Int("edits", 0,
		"reference images an edit may carry, 0 to refuse /v1/images/edits; editing stages the vision "+
			"tower and the VAE encoder (~1.4 GB) and each reference costs ~2.1 GB of prefix KV cache")
	imgCond := flag.Int("image-condition-size", 1024,
		"square whose area every reference image is resized to, and the default output size of an edit "+
			"that names none; diffusers' output_resolution")
	videoOn := flag.Bool("video", false, "load MiniMax-H3 and serve /v1/videos (asynchronous jobs; ~50 GB at a request's peak, nothing staged at rest)")
	videoModel := flag.String("video-model", "models/MiniMax-H3", "MiniMax-H3 checkpoint root")
	videoDir := flag.String("video-dir", "", "where finished videos are kept until they expire; empty is a temporary directory")
	videoTTL := flag.Duration("video-ttl", 24*time.Hour, "how long a finished video job and its file are kept")
	videoQueue := flag.Int("video-queue", 16, "video jobs that may wait behind the running one")
	videoPrompt := flag.Int("video-max-prompt", 0, "longest video prompt in tokens; 0 is the pipeline's 4096")
	flag.Parse()

	if !*tts && !*stt && !*llmOn && !*embedOn && !*imgOn && !*kevOn && !*videoOn {
		log.Printf("warning: no model was asked for; every endpoint will answer 501. " +
			"Pass -llm, -embed, -image, -tts and/or -stt.")
	}
	imgW, imgH, err := parseSize(*imgSize)
	if *imgOn && err != nil {
		log.Fatalf("-image-size: %v", err)
	}

	// **P3a/D19: the server stages the shipped widths unless told otherwise.**
	// `llm.DenseBankPlan` reads `LLM_DENSE_BANK` and defaults to off, because
	// a measurement tool that staged a plan nobody named would make every CSV
	// in `results/` ambiguous. A *product* run wants the opposite default, so
	// the opt-in is here and nowhere else: set the variable to any other plan
	// to override it, or to `off` to serve L8a's int8 bank.
	// The presets are read before anything is staged, so a typo in the file
	// fails in a second rather than after the model's minute of loading.
	var presets []api.Preset
	if *llmOn && *llmPresets != "" {
		explicit := false
		flag.Visit(func(f *flag.Flag) { explicit = explicit || f.Name == "llm-presets" })
		ps, err := api.LoadPresets(*llmPresets)
		switch {
		case err == nil:
			presets = ps
			names := make([]string, len(ps))
			for i, p := range ps {
				names[i] = p.Name
			}
			log.Printf("llm: presets %s from %s", strings.Join(names, ", "), *llmPresets)
		case errors.Is(err, os.ErrNotExist) && !explicit:
			log.Printf("llm: no presets (%s does not exist)", *llmPresets)
		default:
			log.Fatal(err)
		}
	}
	if *llmOn {
		if _, named := os.LookupEnv("LLM_DENSE_BANK"); !named {
			if err := os.Setenv("LLM_DENSE_BANK", llm.ShippedDenseBank); err != nil {
				log.Fatalf("setting the shipped dense bank: %v", err)
			}
		}
		log.Printf("llm: dense bank %s", llm.DenseBankPlan())
		// **P4b/D20**, the same opt-in one block over: four MoE rows that
		// ship at 8.5 bits and have a format the kernels already read.
		// 0.129 GB a token and 1.41 GB of residency for a perplexity delta
		// the instrument cannot resolve.
		if _, named := os.LookupEnv("LLM_MOE_BANK"); !named {
			if err := os.Setenv("LLM_MOE_BANK", llm.ShippedMoEBank); err != nil {
				log.Fatalf("setting the shipped MoE bank: %v", err)
			}
		}
		log.Printf("llm: moe bank %s", llm.MoEBankPlanFromEnv())
		// **P4c/D21 needs one more thing than a plan.** `down_exps=iq4_nl`
		// re-fits 40 billion weights through a sixteen-scale search, which is
		// about eight minutes of this machine — fine for a measurement run
		// and not for a server that otherwise stages in 34 seconds. The
		// transcode is a pure function of (tensor, format, arm), so the
		// server keeps it beside the checkpoint and pays it once ever.
		// `LLM_BANK_CACHE=` (empty) turns it off; `cmd/llm` never sets it.
		if _, named := os.LookupEnv("LLM_BANK_CACHE"); !named {
			dir := *llmModel
			if st, err := os.Stat(dir); err == nil && !st.IsDir() {
				dir = filepath.Dir(dir)
			}
			cache := filepath.Join(dir, "bank-cache")
			if err := os.Setenv("LLM_BANK_CACHE", cache); err != nil {
				log.Fatalf("setting the bank cache: %v", err)
			}
			log.Printf("llm: bank cache %s (first start writes it; delete it to re-fit)", cache)
		}
	}

	srv := &api.Server{Token: *token, MaxUploadBytes: *maxUpload << 20, LogBodies: *logBodies, Presets: presets}

	// The device is opened once and shared. Only the models that were asked
	// for decide whether it is needed at all: a CPU-only run should not fail
	// on a machine without Vulkan.
	var dev *backend.Device
	if *gpu && (*stt || *llmOn || *embedOn || *imgOn || *kevOn || *videoOn || (*tts && *ttsGPU)) {
		d, err := backend.OpenDevice("strix-halo-serve")
		if err != nil {
			log.Fatalf("opening the device: %v", err)
		}
		defer d.Close()
		dev = d
		log.Printf("device: %s", d.Name())
	}

	if *tts {
		start := time.Now()
		ttsDev := dev
		if !*ttsGPU {
			ttsDev = nil
		}
		b, err := backend.NewTTS(backend.TTSOptions{
			Model: *ttsModel, Lexicon: *lexicon, Espeak: *espeak, British: *british,
			Voice: *voice, Device: ttsDev, Noise: *noise, MaxFrames: *ttsFrames,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		srv.Speech = b
		log.Printf("tts: %s, %d voices, %s, in %v", *ttsModel, len(b.Voices()),
			where(ttsDev != nil), time.Since(start).Round(time.Millisecond))
	}

	if *stt {
		start := time.Now()
		b, err := backend.NewSTT(backend.STTOptions{
			Model: *sttModel, Device: dev, MaxSeconds: *maxAudio,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		srv.Transcription = b
		log.Printf("stt: %s, clips to %.0f s, %s, in %v", *sttModel, *maxAudio,
			where(dev != nil), time.Since(start).Round(time.Millisecond))
	}

	if *embedOn {
		start := time.Now()
		b, err := backend.NewEmbed(backend.EmbedOptions{
			Model: *embedModel, Device: dev, MaxTokens: *embedTokens,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		srv.Embedding = b
		log.Printf("embed: %s, inputs to %d tokens, %s, in %v", *embedModel, b.MaxTokens(),
			where(dev != nil), time.Since(start).Round(time.Millisecond))
	}

	if *kevOn {
		if dev == nil {
			log.Fatal("-kev needs the device; it has no CPU path")
		}
		start := time.Now()
		b, err := backend.NewKev(backend.KevOptions{
			Model: *kevModel, Base: *kevBase, Device: dev, MaxTokens: *kevTokens, FP16: *kevFP16,
			CacheStates: cacheStates(*kevCache), CacheTokens: *kevCacheTokens, MaxBatch: *kevBatch,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		srv.SystemOne = b
		slots, cached := b.Cache()
		log.Printf("kev: %s on %s, %s weights, passes of %d tokens, %d cached states of up to %d tokens, in %v",
			*kevModel, *kevBase, b.Bank(), b.MaxTokens(), slots, cached,
			time.Since(start).Round(time.Millisecond))
	}

	if *imgOn {
		start := time.Now()
		b, err := backend.NewImage(backend.ImageOptions{
			Model: *imgModel, Device: dev, Width: imgW, Height: imgH,
			Steps: *imgSteps, MaxPrompt: *imgPrompt, Refs: *imgEdits, CondSize: *imgCond,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		srv.Image = b
		enc, tr, vaeW, act := b.Residency()
		geo := b.Geometry()
		previews := "no previews"
		if geo.Previews {
			previews = fmt.Sprintf("previews to %d partial images", geo.MaxPartials)
		}
		edits := "no edits (start with -edits N)"
		if geo.Edits {
			ref := "reference images"
			if geo.MaxRefs == 1 {
				ref = "reference image"
			}
			edits = fmt.Sprintf("edits with up to %d %s at %d², ", geo.MaxRefs, ref, *imgCond)
			ew, ec := b.EditResidency()
			edits += fmt.Sprintf("%.1f GB + %.1f GB of prefix cache", float64(ew)/1e9, float64(ec)/1e9)
		}
		log.Printf("image: %s, to %dx%d, %d steps, %s, %s, %.1f GB (%.1f encoder + %.1f transformer + %.1f vae + %.1f activations), in %v",
			*imgModel, imgW, imgH, *imgSteps, previews, edits,
			float64(enc+tr+vaeW+act)/1e9, float64(enc)/1e9, float64(tr)/1e9, float64(vaeW)/1e9, float64(act)/1e9,
			time.Since(start).Round(time.Millisecond))
	}

	if *videoOn {
		if dev == nil {
			log.Fatal("-video needs the device; it has no host path (drop -gpu=false)")
		}
		start := time.Now()
		b, err := backend.NewVideo(backend.VideoOptions{Model: *videoModel, Device: dev, MaxPrompt: *videoPrompt})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		jobs, err := api.NewVideoJobs(b, api.VideoJobsOptions{Dir: *videoDir, TTL: *videoTTL, MaxQueued: *videoQueue})
		if err != nil {
			log.Fatal(err)
		}
		// Deferred after the backend's Close, so it runs first: the running
		// job is cancelled and waited for before anything is freed.
		defer jobs.Close()
		srv.Videos = jobs
		geo := b.VideoGeometry()
		log.Printf("video: %s, jobs in %s (kept %v, %d queued at most), default %s x %gs x %d steps, in %v",
			*videoModel, jobs.Dir(), *videoTTL, *videoQueue, geo.DefaultSize, geo.DefaultSeconds, geo.DefaultSteps,
			time.Since(start).Round(time.Millisecond))
	}

	// The Wyoming door is built here, between the speech backends and the
	// language model, and for the same reason the language model is last: a
	// misspelt voice or an address something else already holds should be
	// reported in the first second rather than after 84 GB of staging. The
	// listener is opened now and served later.
	var (
		wy   *wyoming.Server
		wyLn net.Listener
	)
	if *wyAddr != "" {
		w, err := wyoming.New(wyoming.Options{
			Speech: srv.Speech, Transcription: srv.Transcription,
			Name: *wyName, Voice: *voice, AllVoices: *wyAllVoices,
			MaxSeconds: *maxAudio, OutputRate: *wyRate, MaxConns: *wyMaxConns,
		})
		if err != nil {
			log.Fatal(err)
		}
		ln, err := net.Listen("tcp", *wyAddr)
		if err != nil {
			log.Fatalf("wyoming: listening on %s: %v", *wyAddr, err)
		}
		wy, wyLn = w, ln
	}

	// The language model is staged last, and on purpose: it is most of this
	// machine's memory and tens of seconds of staging, so a mistake in a
	// speech flag is reported before the wait rather than after it.
	if *llmOn {
		start := time.Now()
		b, err := backend.NewLLM(backend.LLMOptions{
			Model: *llmModel, Device: dev, Context: *llmCtx, Batch: *llmBatch,
			MaxTokens: *llmMax, Layers: *llmLayers,
			Slots: *llmSlots, PreemptChunk: *llmPreempt, Reserve: *llmReserve, Class: *llmClass,
			NoCheckpoints: !*llmCheckpoints, NoBatchDecode: !*llmBatchDecode, Progress: *llmProgress,
			MMProj: *llmMMProj, VisionTokens: *llmVisionTokens, VisionImages: *llmVisionImages,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer b.Close()
		srv.Completion = b
		layers := "every layer"
		if *llmLayers > 0 {
			layers = fmt.Sprintf("the first %d layers only", *llmLayers)
		}
		vision := "text only"
		if *llmMMProj != "" {
			vision = fmt.Sprintf("images up to %d tokens, %d a request", *llmVisionTokens, *llmVisionImages)
		}
		log.Printf("llm: %s, %s, %d slots of %d cells of context, %d-token batches, %s, in %v",
			*llmModel, layers, b.Slots(), b.Context(), *llmBatch, vision, time.Since(start).Round(time.Millisecond))
	}

	// Say which it is, every time. Only logging the open case meant the
	// usual case -- a token is in force, because one is the default -- was
	// invisible at startup, and the first sign of it was a 401 from a
	// client that never knew it had to send anything.
	if *token == "" {
		// An open server on loopback is a development convenience. An open
		// server on a routable address is every model on this box offered
		// to whoever can reach the port, so say which of the two this is
		// rather than printing one line for both.
		if openToNetwork(*addr) {
			log.Printf("auth: -token is empty and -addr is %s: "+
				"this server authorizes nothing and is reachable from the network. "+
				"Anyone who can reach this port can run the models on it.", *addr)
		} else {
			log.Printf("auth: -token is empty; this server authorizes nothing, "+
				"and -addr %s is loopback, so only this machine can reach it", *addr)
		}
	} else {
		log.Printf("auth: bearer token required on every /v1 request; "+
			"send %q (set it with -token, or -token= to serve open)",
			"Authorization: Bearer "+redactToken(*token))
	}
	// Wyoming is unauthenticated by design -- the protocol has no place to
	// put a credential -- so the token line above does not describe it and
	// this one has to.
	if wy != nil {
		info := wy.Info()
		log.Printf("wyoming: %d asr, %d tts (%d voices) as %q on %s; "+
			"the protocol carries no credential, so -token does not apply to this port",
			len(info.ASR), len(info.TTS), wyomingVoices(info), *wyName, wyLn.Addr())
	}

	if err := listen(srv, *addr, wy, wyLn); err != nil {
		log.Fatal(err)
	}
}

// wyomingVoices counts what the Wyoming door will offer, for the startup
// line. It is a count rather than a list because 28 voice names is a screen
// of journal and GET /v1/models already prints them.
func wyomingVoices(info *wyoming.Info) int {
	n := 0
	for _, p := range info.TTS {
		n += len(p.Voices)
	}
	return n
}

// redactToken shows a token's first and last two characters so the startup
// line can be matched against what a client is configured with, without
// putting the credential in the journal. Anything short enough that this
// would give most of it away is shown as its length alone.
func redactToken(tok string) string {
	if len(tok) < 12 {
		return fmt.Sprintf("<%d characters>", len(tok))
	}
	return tok[:2] + strings.Repeat(".", 6) + tok[len(tok)-2:]
}

// openToNetwork reports whether -addr is reachable from somewhere other than
// this machine. An empty host ("" in ":11434") means every interface, which
// is the easy one to miss.
func openToNetwork(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		// Not a host:port at all; say nothing reassuring about it.
		return true
	}
	if host == "" {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return !strings.EqualFold(host, "localhost")
	}
	return !ip.IsLoopback()
}

// parseSize reads -image-size, which is spelled the way a request spells it so
// that the flag and the API field are not two notations for one thing.
func parseSize(s string) (int, int, error) {
	w, h, ok := strings.Cut(strings.ToLower(strings.TrimSpace(s)), "x")
	if !ok {
		return 0, 0, fmt.Errorf("%q is not WIDTHxHEIGHT", s)
	}
	width, err := strconv.Atoi(strings.TrimSpace(w))
	if err != nil || width <= 0 {
		return 0, 0, fmt.Errorf("%q: %q is not a width", s, w)
	}
	height, err := strconv.Atoi(strings.TrimSpace(h))
	if err != nil || height <= 0 {
		return 0, 0, fmt.Errorf("%q: %q is not a height", s, h)
	}
	return width, height, nil
}

func where(gpu bool) string {
	if gpu {
		return "on the device"
	}
	return "on the host"
}

// listen serves until the process is interrupted, then drains.
//
// The drain matters here more than it does for a CRUD server: a request in
// flight owns the GPU queue, and killing the process underneath it would
// leave the device to the driver to clean up. Shutdown waits for the handler
// to finish, and the deferred backend teardown in main then runs with nothing
// dispatching.
//
// Both doors drain on the same signal and both are waited for, because both
// dispatch to the same device: returning while a Wyoming client is still
// mid-utterance would tear the backends down underneath it exactly as
// killing an HTTP request would. A failure on either listener stops the
// process, since a server that is half the thing it was started as is worse
// than one that says why it stopped.
func listen(srv *api.Server, addr string, wy *wyoming.Server, wyLn net.Listener) error {
	hs := srv.HTTPServer(addr)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 2)
	go func() {
		log.Printf("listening on http://%s", addr)
		for _, route := range []string{
			"/v1/models", "/v1/chat/completions", "/v1/embeddings",
			"/v1/audio/speech", "/v1/audio/transcriptions",
			"/v1/images/generations", "/v1/images/edits", "/v1/systemone",
			"/v1/videos",
		} {
			log.Printf("  %s", route)
		}
		errs <- hs.ListenAndServe()
	}()

	// wyDone is closed when Serve has returned, which is what tells the
	// shutdown path that every connection has finished rather than merely
	// that the listener is shut.
	wyDone := make(chan struct{})
	if wy != nil {
		go func() {
			defer close(wyDone)
			log.Printf("listening on tcp://%s (wyoming)", wyLn.Addr())
			if err := wy.Serve(ctx, wyLn); err != nil {
				errs <- err
			}
		}()
	} else {
		close(wyDone)
	}

	select {
	case err := <-errs:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		log.Printf("shutting down; waiting for requests in flight")
		shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		err := hs.Shutdown(shutdown)
		// Serve already saw the same cancelled context and is unwinding;
		// what is left is to wait for the utterance it may be in the
		// middle of.
		select {
		case <-wyDone:
		case <-shutdown.Done():
			log.Printf("wyoming: connections still open after 30 s; closing anyway")
		}
		if err != nil {
			return fmt.Errorf("draining: %w", err)
		}
		return nil
	}
}

// cacheStates maps -kev-cache onto backend.KevOptions, where zero means the
// default: 0 on the command line is "off".
func cacheStates(n int) int {
	if n <= 0 {
		return -1
	}
	return n
}
