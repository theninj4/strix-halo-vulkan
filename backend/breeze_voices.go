package backend

// The custom voice store (research/tts-instruct-vertical.md B8): consents and
// voices as files under one directory, so they outlive the process.
//
//	<dir>/consents/cons_<hex>.json   the record; .wav beside it, the recording
//	<dir>/voices/voice_<hex>.json    the record, the transcript and the codes
//
// A voice is its reference codes (the codec encoder's, on the host, ~0.25 s
// a second of sample) and the sample's exact transcript: what a clone's
// prompt is built from. The sample's audio is not kept; the codes are it.

import (
	"context"
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"strix-halo-vulkan/api"
	"strix-halo-vulkan/audio"
	"strix-halo-vulkan/breeze"
)

type storedVoice struct {
	api.Voice
	Codes [][16]int32 `json:"codes"`
}

type voiceStore struct {
	dir   string
	model string
	mu    sync.Mutex
	enc   *breeze.CodecEncoder // loaded on the first voice
	byID  map[string]*storedVoice
}

func openVoiceStore(dir, model string) (*voiceStore, error) {
	for _, sub := range []string{"consents", "voices"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			return nil, err
		}
	}
	s := &voiceStore{dir: dir, model: model, byID: map[string]*storedVoice{}}
	files, err := filepath.Glob(filepath.Join(dir, "voices", "voice_*.json"))
	if err != nil {
		return nil, err
	}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		var v storedVoice
		if err := json.Unmarshal(b, &v); err != nil {
			return nil, fmt.Errorf("backend: voice %s: %w", f, err)
		}
		s.byID[v.ID] = &v
	}
	return s, nil
}

func newID(prefix string) string {
	var b [12]byte
	_, _ = crand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}

func writeAtomic(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (s *voiceStore) get(id string) *storedVoice {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.byID[id]
}

func (s *voiceStore) list() []api.Voice {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]api.Voice, 0, len(s.byID))
	for _, v := range s.byID {
		out = append(out, v.Voice)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt < out[j].CreatedAt })
	return out
}

func (b *Breeze) needStore() (*voiceStore, error) {
	if b.store == nil {
		return nil, fmt.Errorf("this server keeps no custom voices (-tts-instruct-voices): %w", api.ErrUnsupported)
	}
	return b.store, nil
}

// CreateConsent keeps a consent recording the HTTP layer has heard.
func (b *Breeze) CreateConsent(_ context.Context, name, language string, rec *audio.Clip) (*api.VoiceConsent, error) {
	s, err := b.needStore()
	if err != nil {
		return nil, err
	}
	c := &api.VoiceConsent{ID: newID("cons_"), Object: "audio.voice_consent", Name: name, Language: language,
		CreatedAt: time.Now().Unix()}
	base := filepath.Join(s.dir, "consents", c.ID)
	if err := rec.WriteWAV(base + ".wav"); err != nil {
		return nil, err
	}
	j, _ := json.MarshalIndent(c, "", " ")
	if err := writeAtomic(base+".json", j); err != nil {
		return nil, err
	}
	return c, nil
}

// CreateVoice encodes a sample into a voice under a stored consent.
func (b *Breeze) CreateVoice(_ context.Context, name string, sample *audio.Clip, transcript, consent string) (*api.Voice, error) {
	s, err := b.needStore()
	if err != nil {
		return nil, err
	}
	if !strings.HasPrefix(consent, "cons_") || strings.ContainsAny(consent, "/\\.") {
		return nil, fmt.Errorf("no consent %q: %w", consent, api.ErrUnsupported)
	}
	if _, err := os.Stat(filepath.Join(s.dir, "consents", consent+".json")); err != nil {
		return nil, fmt.Errorf("no consent %q (POST /v1/audio/voice_consents first): %w", consent, api.ErrUnsupported)
	}
	c24, err := audio.Resample(sample, breeze.SampleRate)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.enc == nil {
		if s.enc, err = breeze.LoadCodecEncoder(s.model); err != nil {
			s.mu.Unlock()
			return nil, err
		}
	}
	enc := s.enc
	s.mu.Unlock()
	codes, err := enc.Encode(c24.Samples, nil)
	if err != nil {
		return nil, err
	}
	v := &storedVoice{Voice: api.Voice{ID: newID("voice_"), Object: "audio.voice", Name: name, Consent: consent,
		CreatedAt: time.Now().Unix(), Transcript: strings.TrimSpace(transcript), Seconds: sample.Duration()}, Codes: codes}
	j, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	if err := writeAtomic(filepath.Join(s.dir, "voices", v.ID+".json"), j); err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.byID[v.ID] = v
	s.mu.Unlock()
	return &v.Voice, nil
}

// ListVoices is every stored voice, oldest first.
func (b *Breeze) ListVoices() []api.Voice {
	if b.store == nil {
		return []api.Voice{}
	}
	return b.store.list()
}

// DeleteVoice removes a voice.
func (b *Breeze) DeleteVoice(id string) error {
	s, err := b.needStore()
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.byID[id] == nil {
		return api.ErrNotFound
	}
	if err := os.Remove(filepath.Join(s.dir, "voices", id+".json")); err != nil {
		return err
	}
	delete(s.byID, id)
	return nil
}
