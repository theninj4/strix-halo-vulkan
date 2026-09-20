// Package wyoming serves the two speech verticals over the Wyoming protocol,
// which is what Home Assistant's voice pipeline speaks.
//
// It is a second door onto the same backends `api` serves over HTTP, and it
// exists because Home Assistant does not speak OpenAI's audio API: its
// `wyoming` integration connects to a TCP port, asks what is behind it, and
// then streams microphone audio in and synthesised speech out. Nothing in
// here owns a model or touches Vulkan -- a Server holds the same
// api.SpeechBackend and api.TranscriptionBackend the HTTP handlers do, so the
// two protocols are one process, one staging of the weights and one GPU
// queue.
//
// The wire format is a newline-terminated JSON header, then an optional JSON
// data block, then an optional binary payload:
//
//	{"type":"audio-chunk","version":"1.10.2","data_length":57,"payload_length":2048}\n
//	{"rate": 16000, "width": 2, "channels": 1, "timestamp": 0}
//	<2048 bytes of little-endian signed 16-bit PCM>
//
// The lengths are what makes it framed rather than line-oriented: the payload
// is raw PCM and may contain a newline. `data` may also appear inline in the
// header, in which case the trailing block is merged over it, which is what
// the reference implementation does and what ReadEvent reproduces.
//
// This file is the framing alone. info.go is what the protocol's events look
// like, voices.go is how a kokoro voice pack is described to a client, and
// server.go is the conversation.
package wyoming

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// protocolVersion is the `version` every event this server writes carries.
//
// It is the version of the `wyoming` Python package whose wire format this
// implements, rather than a version of this server: the field is there so a
// client can tell what the framing is, and answering with our own release
// number would tell it nothing it could act on. No client this has been run
// against reads it.
const protocolVersion = "1.10.2"

// The three bounds an event is read under. They exist because this is a
// protocol whose frames declare their own length: a header that says
// "payload_length: 4294967296" is one allocation away from the process, and
// the only defence is to disbelieve it.
//
// maxPayloadBytes is the loosest because the payload is audio and a client is
// free to send a whole utterance in one chunk. It is still far above what any
// real one does -- Home Assistant sends 1024 samples at a time, 2 KB.
const (
	maxHeaderBytes  = 64 << 10
	maxDataBytes    = 1 << 20
	maxPayloadBytes = 16 << 20
)

// Event is one message: a type, an optional JSON object and an optional
// binary payload.
//
// Data is kept as raw JSON rather than a map because every consumer here
// wants it as a struct, and decoding it into `map[string]any` first would
// turn every timestamp into a float64 and every field name into a runtime
// string lookup. Unmarshal is the only way it is read.
type Event struct {
	Type    string
	Data    json.RawMessage
	Payload []byte
}

// Unmarshal decodes the event's data into v.
//
// An event with no data is not an error: `describe`, `audio-stop` and
// `synthesize-stop` all carry none, and their typed forms are the zero value.
func (e *Event) Unmarshal(v any) error {
	if len(e.Data) == 0 {
		return nil
	}
	if err := json.Unmarshal(e.Data, v); err != nil {
		return fmt.Errorf("wyoming: %s: %w", e.Type, err)
	}
	return nil
}

// event builds one from a value that marshals to the event's data object.
// A nil data value writes a header and nothing else.
func event(typ string, data any) (*Event, error) {
	ev := &Event{Type: typ}
	if data == nil {
		return ev, nil
	}
	raw, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("wyoming: encoding %s: %w", typ, err)
	}
	ev.Data = raw
	return ev, nil
}

// header is the JSON line at the front of every event.
//
// The lengths are pointers so that "absent" and "zero" stay distinguishable
// on the way in; on the way out they are only set when there is something to
// measure, which is what the reference implementation writes.
type header struct {
	Type          string          `json:"type"`
	Version       string          `json:"version,omitempty"`
	Data          json.RawMessage `json:"data,omitempty"`
	DataLength    *int            `json:"data_length,omitempty"`
	PayloadLength *int            `json:"payload_length,omitempty"`
}

// ReadEvent reads one event, blocking until it has all of it.
//
// io.EOF means the peer closed cleanly between events, which is how every
// Home Assistant request ends and is not a failure. A short read in the
// middle of a declared length is io.ErrUnexpectedEOF and is.
func ReadEvent(r *bufio.Reader) (*Event, error) {
	line, err := readHeaderLine(r)
	if err != nil {
		return nil, err
	}
	var h header
	if err := json.Unmarshal(line, &h); err != nil {
		return nil, fmt.Errorf("wyoming: malformed event header %s: %w", truncate(line), err)
	}
	if h.Type == "" {
		return nil, fmt.Errorf("wyoming: event header has no type: %s", truncate(line))
	}
	ev := &Event{Type: h.Type, Data: h.Data}

	if n := h.DataLength; n != nil && *n > 0 {
		if *n > maxDataBytes {
			return nil, fmt.Errorf("wyoming: %s declares %d bytes of data, over the %d-byte limit",
				h.Type, *n, maxDataBytes)
		}
		buf := make([]byte, *n)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("wyoming: %s: reading %d bytes of data: %w", h.Type, *n, err)
		}
		if ev.Data, err = mergeData(h.Data, buf); err != nil {
			return nil, fmt.Errorf("wyoming: %s: %w", h.Type, err)
		}
	}

	if n := h.PayloadLength; n != nil && *n > 0 {
		if *n > maxPayloadBytes {
			return nil, fmt.Errorf("wyoming: %s declares %d bytes of payload, over the %d-byte limit",
				h.Type, *n, maxPayloadBytes)
		}
		ev.Payload = make([]byte, *n)
		if _, err := io.ReadFull(r, ev.Payload); err != nil {
			return nil, fmt.Errorf("wyoming: %s: reading %d bytes of payload: %w", h.Type, *n, err)
		}
	}
	return ev, nil
}

// WriteEvent writes one event and flushes it.
//
// The flush is not optional: every event this server sends is either an
// answer somebody is blocked on or a chunk of audio somebody is playing, so
// there is never a reason to hold one in a buffer waiting for the next.
func WriteEvent(w *bufio.Writer, ev *Event) error {
	h := header{Type: ev.Type, Version: protocolVersion}
	data := ev.Data
	if isEmptyObject(data) {
		data = nil
	}
	if len(data) > 0 {
		n := len(data)
		h.DataLength = &n
	}
	if len(ev.Payload) > 0 {
		n := len(ev.Payload)
		h.PayloadLength = &n
	}
	line, err := json.Marshal(h)
	if err != nil {
		return fmt.Errorf("wyoming: encoding the %s header: %w", ev.Type, err)
	}
	if _, err := w.Write(line); err != nil {
		return err
	}
	if err := w.WriteByte('\n'); err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		return err
	}
	if _, err := w.Write(ev.Payload); err != nil {
		return err
	}
	return w.Flush()
}

// readHeaderLine reads up to and including the next newline, bounded.
//
// bufio's own ReadBytes would do this in one call and would also let a peer
// that never sends a newline grow the heap until the process dies, which is
// the whole reason this is written out: the loop stops at maxHeaderBytes.
// Blank lines are skipped rather than refused -- they are not legal, but
// tolerating them costs nothing and a stray newline should not drop a
// satellite's connection.
func readHeaderLine(r *bufio.Reader) ([]byte, error) {
	var line []byte
	for {
		chunk, err := r.ReadSlice('\n')
		line = append(line, chunk...)
		// Checked before the error is classified, so that the bound holds
		// however the read ended rather than only while the peer keeps
		// sending.
		if len(line) > maxHeaderBytes {
			return nil, fmt.Errorf("wyoming: event header is over %d bytes with no newline",
				maxHeaderBytes)
		}
		switch {
		case err == nil:
			if trimmed := trimSpace(line); len(trimmed) > 0 {
				return trimmed, nil
			}
			line = line[:0] // a blank line; keep reading
		case errors.Is(err, bufio.ErrBufferFull):
			// More of the same line is coming.
		case errors.Is(err, io.EOF):
			if len(trimSpace(line)) == 0 {
				// A clean close between events, which is how a request
				// ends. Trailing whitespace closes just as cleanly.
				return nil, io.EOF
			}
			// Half a header: the peer stopped mid-line.
			return nil, fmt.Errorf("wyoming: %d bytes of event header with no newline: %w",
				len(line), io.ErrUnexpectedEOF)
		default:
			return nil, err
		}
	}
}

// mergeData overlays the trailing data block on whatever the header carried
// inline. The reference implementation writes one or the other and never
// both, so the common path is the early return; the merge is here because the
// reference *reader* merges, and a client written against it is entitled to.
func mergeData(inline, trailing []byte) (json.RawMessage, error) {
	if isEmptyObject(inline) {
		return trailing, nil
	}
	var base, over map[string]json.RawMessage
	if err := json.Unmarshal(inline, &base); err != nil {
		return nil, fmt.Errorf("inline data: %w", err)
	}
	if err := json.Unmarshal(trailing, &over); err != nil {
		return nil, fmt.Errorf("trailing data: %w", err)
	}
	for k, v := range over {
		base[k] = v
	}
	merged, err := json.Marshal(base)
	if err != nil {
		return nil, err
	}
	return merged, nil
}

// isEmptyObject reports whether raw is absent, null or `{}`. All three mean
// the same thing to this protocol -- an event with no data -- and the
// reference implementation writes no data block for any of them.
func isEmptyObject(raw []byte) bool {
	s := trimSpace(raw)
	return len(s) == 0 || string(s) == "null" || string(s) == "{}"
}

// trimSpace is bytes.TrimSpace for the ASCII whitespace JSON allows, written
// here so that this file imports the JSON codec and nothing else.
func trimSpace(b []byte) []byte {
	i, j := 0, len(b)
	for i < j && isSpace(b[i]) {
		i++
	}
	for j > i && isSpace(b[j-1]) {
		j--
	}
	return b[i:j]
}

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\r' || c == '\n' }

// truncate bounds a malformed header on its way into an error message. A
// client that sent 64 KB of garbage should not put 64 KB of it in the log.
func truncate(b []byte) string {
	const max = 200
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max]) + "..."
}
