package wyoming

// The event types this server reads and writes. Wyoming has more of them --
// wake word detection, intent handling, satellite control -- and the rule for
// a type that is not here is to drop it silently rather than to refuse it,
// which is what the protocol asks of a server and what lets a newer client
// talk to an older one.
const (
	typeDescribe      = "describe"
	typeInfo          = "info"
	typePing          = "ping"
	typePong          = "pong"
	typeError         = "error"
	typeSelectProgram = "select-program"

	typeTranscribe = "transcribe"
	typeTranscript = "transcript"

	typeSynthesize = "synthesize"

	typeAudioStart = "audio-start"
	typeAudioChunk = "audio-chunk"
	typeAudioStop  = "audio-stop"
)

// Attribution is who made an artifact and where it came from. Every artifact
// in an `info` message carries one, and the reference client's decoder
// requires it -- an artifact without it fails to construct, rather than
// arriving with an empty one.
type Attribution struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

// Artifact is what a program and a model have in common.
//
// Description and Version are pointers so that "we do not have one" is sent
// as an explicit null rather than by leaving the field out. The reference
// decoder treats those two the same way, but a null says the server
// considered the question and a missing key does not.
type Artifact struct {
	Name        string      `json:"name"`
	Description *string     `json:"description"`
	Attribution Attribution `json:"attribution"`
	Installed   bool        `json:"installed"`
	Version     *string     `json:"version"`
}

// AsrModel is one speech-to-text model and the languages it hears.
type AsrModel struct {
	Artifact
	Languages []string `json:"languages"`
}

// AsrProgram is a speech-to-text service.
//
// The three `prefers`/`requires` flags are advice to a satellite about the
// audio it sends, and they are answered from what this checkpoint actually
// is. **RequiresExternalVAD is true**: parakeet transcribes a clip that has
// already been cut, and nothing in this repository decides where a voice
// command ended, so the satellite or Home Assistant has to. The other two are
// true for the ordinary reason -- a far-field microphone's gain and noise are
// a room's problem and not a model's.
type AsrProgram struct {
	Artifact
	Models                       []AsrModel `json:"models"`
	SupportsTranscriptStreaming  bool       `json:"supports_transcript_streaming"`
	RequiresExternalVAD          bool       `json:"requires_external_vad"`
	PrefersAutoGainEnabled       bool       `json:"prefers_auto_gain_enabled"`
	PrefersNoiseReductionEnabled bool       `json:"prefers_noise_reduction_enabled"`
}

// TtsVoiceSpeaker is one speaker inside a multi-speaker voice. Kokoro has
// none: a voice pack is one speaker, and a blend of packs is spelled in the
// voice name rather than selected here, so this type exists to be absent.
type TtsVoiceSpeaker struct {
	Name string `json:"name"`
}

// TtsVoice is one voice pack. Home Assistant shows Description in its voice
// menu and sends Name back as the voice id, so the two are a readable label
// and the checkpoint's own filename respectively.
type TtsVoice struct {
	Artifact
	Languages []string          `json:"languages"`
	Speakers  []TtsVoiceSpeaker `json:"speakers"`
}

// TtsProgram is a text-to-speech service.
//
// **SupportsSynthesizeStreaming is false**, and that is a measurement rather
// than a gap. Streaming synthesis in this protocol means the client sends the
// text in pieces as it is generated and the server answers each piece with
// audio; what it buys is the time between the first piece and the last. On
// this part kokoro synthesises nineteen seconds of speech in 162 ms
// (SPEECH.md T10), so the whole utterance is ready before a satellite could
// have finished playing the first sentence of a streamed one, and the
// streaming path would only add a way for the pieces to disagree about
// prosody -- the durations are decided for the whole utterance at once.
type TtsProgram struct {
	Artifact
	Voices                      []TtsVoice `json:"voices"`
	SupportsSynthesizeStreaming bool       `json:"supports_synthesize_streaming"`
}

// Info is the answer to `describe`: what this process loaded.
//
// The five empty lists are not padding. The reference decoder iterates each
// of them, so a missing key is fine (it defaults to empty) but a null is a
// crash on the client, and the honest way to say "this server has no wake
// word detector" is an empty list rather than silence.
type Info struct {
	ASR    []AsrProgram `json:"asr"`
	TTS    []TtsProgram `json:"tts"`
	Handle []any        `json:"handle"`
	Intent []any        `json:"intent"`
	Wake   []any        `json:"wake"`
	Mic    []any        `json:"mic"`
	Snd    []any        `json:"snd"`
}

// describeData, pingData and the rest are the payloads of the events this
// server exchanges. They are small enough that a struct per event is shorter
// than the alternative and makes the field names checkable.

// pingData is `ping` and `pong`: the text, if any, is copied back verbatim.
type pingData struct {
	Text *string `json:"text"`
}

// errorData is the `error` event. Every failure this server has goes back as
// one of these, because the alternative -- closing the connection -- is
// reported by Home Assistant as "connection lost" and says nothing about what
// went wrong.
type errorData struct {
	Text string `json:"text"`
	Code string `json:"code,omitempty"`
}

// selectProgramData names which advertised program should handle the rest of
// the connection.
type selectProgramData struct {
	Name string `json:"name"`
}

// transcribeData opens a speech-to-text request. Only the language is read,
// and only to log it: this checkpoint detects the language and has nothing in
// its graph to condition on a hint, which is the same reason
// api.TranscriptionRequest's language reaches backend.STT and is ignored
// there.
type transcribeData struct {
	Name     string `json:"name"`
	Language string `json:"language"`
}

// transcriptData is the answer.
//
// It carries the text and nothing else. The protocol has a `language` field
// and a TDT transducer does not detect one, so writing the language the
// client asked for would be handing back its own guess as a result.
type transcriptData struct {
	Text string `json:"text"`
}

// audioFormat is the PCM description an `audio-start` and every `audio-chunk`
// carries. Width is in bytes per sample.
type audioFormat struct {
	Rate      int  `json:"rate"`
	Width     int  `json:"width"`
	Channels  int  `json:"channels"`
	Timestamp *int `json:"timestamp"`
}

// audioStopData ends a stream. The timestamp is the stream's length in
// milliseconds.
type audioStopData struct {
	Timestamp *int `json:"timestamp"`
}

// synthesizeData is a text-to-speech request.
type synthesizeData struct {
	Text       string           `json:"text"`
	Voice      *synthesizeVoice `json:"voice"`
	TextFormat string           `json:"text_format"`
}

// synthesizeVoice is how a request names a voice: by name, or by language and
// let the server pick. Speaker selects within a multi-speaker voice, which
// none of kokoro's are.
type synthesizeVoice struct {
	Name     string `json:"name"`
	Language string `json:"language"`
	Speaker  string `json:"speaker"`
}
