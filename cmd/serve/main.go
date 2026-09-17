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
//	go run ./cmd/serve -tts -gpu=false              # the CPU reference
//
// The endpoints that answer today are GET /v1/models, POST
// /v1/chat/completions, POST /v1/audio/speech and POST
// /v1/audio/transcriptions. The rest are routed and answer 501: /v1/responses
// and /v1/messages are the same model in two more envelopes, embeddings wait
// on there being an embedding model at all, and images on zimage's pipeline
// being wired in.
//
// Two things about the device are worth knowing before reading the timings.
// **Requests are serialised at the queue**, because a vk.Device has one and
// nothing in `vk` is externally synchronised -- the server is concurrent at
// the HTTP layer and serial at the GPU, which is also the only way these
// models make sense on a part with one 236 GB/s bus. And **-tts-gpu stages
// per request**: kokoro's arenas are sized for one utterance's frame count,
// so the device path pays a staging pass every time and is off by default
// until that is measured (see backend.TTS.synthesize).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/backend"
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
	ttsGPU := flag.Bool("tts-gpu", false, "run kokoro on the device; it stages per utterance, see the package comment")
	lexicon := flag.String("lexicon", "models/misaki", "misaki lexicon directory; empty takes phonemes only")
	espeak := flag.Bool("espeak", true, "use espeak-ng for words outside the lexicon")
	british := flag.Bool("british", false, "use the en-GB lexicon and fallback")
	voice := flag.String("voice", "af_heart", "voice a request that names none gets")
	noise := flag.Int64("noise", 0, "seed for the vocoder's excitation noise; 0 leaves it off")

	llmOn := flag.Bool("llm", false, "load qwen3.8-flash-next and serve /v1/chat/completions")
	llmModel := flag.String("llm-model", defaultLLMModel, "GGUF checkpoint, shard or directory")
	llmCtx := flag.Int("llm-ctx", 4096, "cache cells: the longest conversation, prompt plus completion")
	llmBatch := flag.Int("llm-batch", 512, "tokens the prefill arenas hold; a longer prompt is prefilled in chunks of it")
	llmMax := flag.Int("llm-max-tokens", 1024, "tokens a request that names no max_tokens gets")
	llmLayers := flag.Int("llm-layers", 0, "stage only the first N layers; 0 is the model, anything else is a fast start and not an answer")

	stt := flag.Bool("stt", false, "load parakeet-tdt-0.6b-v3 and serve /v1/audio/transcriptions")
	sttModel := flag.String("stt-model", "models/parakeet-tdt-0.6b-v3", "parakeet checkpoint directory")
	maxAudio := flag.Float64("max-audio", 60, "longest clip the transcription arenas are sized for, in seconds")
	flag.Parse()

	if !*tts && !*stt && !*llmOn {
		log.Printf("warning: no model was asked for; every endpoint will answer 501. " +
			"Pass -llm, -tts and/or -stt.")
	}

	srv := &api.Server{Token: *token, MaxUploadBytes: *maxUpload << 20, LogBodies: *logBodies}

	// The device is opened once and shared. Only the models that were asked
	// for decide whether it is needed at all: a CPU-only run should not fail
	// on a machine without Vulkan.
	var dev *backend.Device
	if *gpu && (*stt || *llmOn || (*tts && *ttsGPU)) {
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
			Voice: *voice, Device: ttsDev, Noise: *noise,
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

	// The language model is staged last, and on purpose: it is most of this
	// machine's memory and tens of seconds of staging, so a mistake in a
	// speech flag is reported before the wait rather than after it.
	if *llmOn {
		start := time.Now()
		b, err := backend.NewLLM(backend.LLMOptions{
			Model: *llmModel, Device: dev, Context: *llmCtx, Batch: *llmBatch,
			MaxTokens: *llmMax, Layers: *llmLayers,
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
		log.Printf("llm: %s, %s, %d cells of context, %d-token batches, in %v",
			*llmModel, layers, b.Context(), *llmBatch, time.Since(start).Round(time.Millisecond))
	}

	if *token == "" {
		log.Printf("warning: -token is empty; this server authorizes nothing")
	}
	if err := listen(srv, *addr); err != nil {
		log.Fatal(err)
	}
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
func listen(srv *api.Server, addr string) error {
	hs := srv.HTTPServer(addr)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 1)
	go func() {
		log.Printf("listening on http://%s", addr)
		for _, route := range []string{
			"/v1/models", "/v1/chat/completions", "/v1/audio/speech", "/v1/audio/transcriptions",
		} {
			log.Printf("  %s", route)
		}
		errs <- hs.ListenAndServe()
	}()

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
		if err := hs.Shutdown(shutdown); err != nil {
			return fmt.Errorf("draining: %w", err)
		}
		return nil
	}
}
