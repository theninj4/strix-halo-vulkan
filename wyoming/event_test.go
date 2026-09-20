package wyoming

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

// The framing is the part of this protocol that has to be exactly right: a
// byte out of place in a length and every event after it is garbage, because
// there is no resynchronisation. So these tests are written against frames
// spelled out by hand the way the reference implementation emits them, rather
// than against this package's own writer -- a round trip through one
// implementation would agree with itself whatever it did.

func TestReadEventReferenceFraming(t *testing.T) {
	// One `audio-chunk` as wyoming's async_write_event produces it: the data
	// object is not in the header, it follows the newline, and the payload
	// follows the data.
	payload := []byte{0x01, 0x00, 0xff, 0x7f, 0x0a, 0x00} // a newline in the audio
	data := `{"rate": 16000, "width": 2, "channels": 1, "timestamp": 0}`
	frame := `{"type": "audio-chunk", "version": "1.10.2", "data_length": ` +
		itoa(len(data)) + `, "payload_length": ` + itoa(len(payload)) + "}\n" + data
	r := bufio.NewReader(bytes.NewReader(append([]byte(frame), payload...)))

	ev, err := ReadEvent(r)
	if err != nil {
		t.Fatalf("ReadEvent: %v", err)
	}
	if ev.Type != "audio-chunk" {
		t.Errorf("type = %q, want audio-chunk", ev.Type)
	}
	var f audioFormat
	if err := ev.Unmarshal(&f); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if f.Rate != 16000 || f.Width != 2 || f.Channels != 1 {
		t.Errorf("format = %+v, want 16000/2/1", f)
	}
	if !bytes.Equal(ev.Payload, payload) {
		t.Errorf("payload = %v, want %v", ev.Payload, payload)
	}
	// Nothing left over: the lengths accounted for the whole frame.
	if _, err := ReadEvent(r); !errors.Is(err, io.EOF) {
		t.Errorf("after the frame: %v, want EOF", err)
	}
}

func TestReadEventInlineData(t *testing.T) {
	// A client that puts the data in the header instead, which the reference
	// reader accepts and some hand-written clients send.
	r := bufio.NewReader(strings.NewReader(`{"type":"transcribe","data":{"language":"en"}}` + "\n"))
	ev, err := ReadEvent(r)
	if err != nil {
		t.Fatalf("ReadEvent: %v", err)
	}
	var d transcribeData
	if err := ev.Unmarshal(&d); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if d.Language != "en" {
		t.Errorf("language = %q, want en", d.Language)
	}
}

func TestReadEventMergesData(t *testing.T) {
	// Both at once: the trailing block wins on a key they share, which is
	// what the reference reader's dict.update does.
	trailing := `{"language":"de"}`
	frame := `{"type":"transcribe","data":{"language":"en","name":"parakeet"},"data_length":` +
		itoa(len(trailing)) + "}\n" + trailing
	ev, err := ReadEvent(bufio.NewReader(strings.NewReader(frame)))
	if err != nil {
		t.Fatalf("ReadEvent: %v", err)
	}
	var d transcribeData
	if err := ev.Unmarshal(&d); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if d.Language != "de" {
		t.Errorf("language = %q, want de (the trailing block overrides the header)", d.Language)
	}
	if d.Name != "parakeet" {
		t.Errorf("name = %q, want parakeet (the header's key survives the merge)", d.Name)
	}
}

func TestWriteEventFraming(t *testing.T) {
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	ev, err := event("audio-chunk", audioFormat{Rate: 24000, Width: 2, Channels: 1})
	if err != nil {
		t.Fatalf("event: %v", err)
	}
	ev.Payload = []byte{1, 2, 3, 4}
	if err := WriteEvent(w, ev); err != nil {
		t.Fatalf("WriteEvent: %v", err)
	}

	line, rest, ok := bytes.Cut(buf.Bytes(), []byte("\n"))
	if !ok {
		t.Fatalf("no newline in %q", buf.String())
	}
	var h struct {
		Type          string `json:"type"`
		Version       string `json:"version"`
		Data          any    `json:"data"`
		DataLength    int    `json:"data_length"`
		PayloadLength int    `json:"payload_length"`
	}
	if err := json.Unmarshal(line, &h); err != nil {
		t.Fatalf("header %q: %v", line, err)
	}
	if h.Version != protocolVersion {
		t.Errorf("version = %q, want %q", h.Version, protocolVersion)
	}
	if h.Data != nil {
		t.Errorf("header carries an inline data object as well as a data_length: %v", h.Data)
	}
	if h.DataLength+h.PayloadLength != len(rest) {
		t.Errorf("lengths say %d bytes follow the header, %d did", h.DataLength+h.PayloadLength, len(rest))
	}
	if h.PayloadLength != 4 {
		t.Errorf("payload_length = %d, want 4", h.PayloadLength)
	}
}

func TestWriteEventNoData(t *testing.T) {
	// `describe` and `audio-stop` with no timestamp carry nothing, and the
	// reference writer emits no data block at all for them -- so neither
	// does this one, or the reader on the other side waits for bytes that
	// are not coming.
	for _, data := range []any{nil, struct{}{}, map[string]any{}} {
		var buf bytes.Buffer
		w := bufio.NewWriter(&buf)
		ev, err := event("describe", data)
		if err != nil {
			t.Fatalf("event: %v", err)
		}
		if err := WriteEvent(w, ev); err != nil {
			t.Fatalf("WriteEvent: %v", err)
		}
		if got := bytes.Count(buf.Bytes(), []byte("\n")); got != 1 {
			t.Errorf("data %v: %d newlines in %q, want 1", data, got, buf.String())
		}
		if bytes.Contains(buf.Bytes(), []byte("data_length")) {
			t.Errorf("data %v: header declares a data_length: %q", data, buf.String())
		}
	}
}

func TestEventRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	want := []*Event{
		mustEvent(t, "describe", nil),
		mustEvent(t, "transcript", transcriptData{Text: "and so my fellow Americans"}),
		mustEvent(t, "audio-stop", audioStopData{Timestamp: intptr(11000)}),
	}
	want[1].Payload = nil
	chunk := mustEvent(t, "audio-chunk", audioFormat{Rate: 24000, Width: 2, Channels: 1, Timestamp: intptr(0)})
	chunk.Payload = bytes.Repeat([]byte{0xde, 0xad}, 1024)
	want = append(want, chunk)

	for _, ev := range want {
		if err := WriteEvent(w, ev); err != nil {
			t.Fatalf("WriteEvent %s: %v", ev.Type, err)
		}
	}
	r := bufio.NewReader(bytes.NewReader(buf.Bytes()))
	for _, ev := range want {
		got, err := ReadEvent(r)
		if err != nil {
			t.Fatalf("ReadEvent %s: %v", ev.Type, err)
		}
		if got.Type != ev.Type {
			t.Fatalf("type = %q, want %q", got.Type, ev.Type)
		}
		if !bytes.Equal(got.Payload, ev.Payload) {
			t.Errorf("%s: payload is %d bytes, want %d", ev.Type, len(got.Payload), len(ev.Payload))
		}
		if !jsonEqual(got.Data, ev.Data) {
			t.Errorf("%s: data = %s, want %s", ev.Type, got.Data, ev.Data)
		}
	}
	if _, err := ReadEvent(r); !errors.Is(err, io.EOF) {
		t.Errorf("after the last event: %v, want EOF", err)
	}
}

func TestReadEventBounds(t *testing.T) {
	for _, tt := range []struct {
		name, frame, want string
	}{{
		// A length a peer declares is a peer's number, and believing it is
		// one allocation away from the process.
		"payload", `{"type":"audio-chunk","payload_length":99999999999}` + "\n", "over the",
	}, {
		"data", `{"type":"audio-chunk","data_length":99999999}` + "\n", "over the",
	}, {
		"no type", `{"version":"1.10.2"}` + "\n", "no type",
	}, {
		"not json", "hello\n", "malformed",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ReadEvent(bufio.NewReader(strings.NewReader(tt.frame)))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want one mentioning %q", err, tt.want)
			}
		})
	}
}

func TestReadEventUnboundedHeader(t *testing.T) {
	// A peer that never sends a newline must not be able to grow the heap.
	r := bufio.NewReader(strings.NewReader(`{"type":"` + strings.Repeat("a", maxHeaderBytes+1024)))
	if _, err := ReadEvent(r); err == nil || !strings.Contains(err.Error(), "no newline") {
		t.Errorf("err = %v, want one mentioning the missing newline", err)
	}
}

func TestReadEventTruncated(t *testing.T) {
	// A frame that declares more payload than it has is ErrUnexpectedEOF,
	// not a clean close: it is the difference between "the client hung up"
	// and "the client hung up mid-utterance".
	frame := `{"type":"audio-chunk","payload_length":8}` + "\n" + "abc"
	_, err := ReadEvent(bufio.NewReader(strings.NewReader(frame)))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("err = %v, want ErrUnexpectedEOF", err)
	}
}

func TestReadEventSkipsBlankLines(t *testing.T) {
	r := bufio.NewReader(strings.NewReader("\n\n" + `{"type":"ping"}` + "\n"))
	ev, err := ReadEvent(r)
	if err != nil {
		t.Fatalf("ReadEvent: %v", err)
	}
	if ev.Type != "ping" {
		t.Errorf("type = %q, want ping", ev.Type)
	}
}

func mustEvent(t *testing.T, typ string, data any) *Event {
	t.Helper()
	ev, err := event(typ, data)
	if err != nil {
		t.Fatalf("event %s: %v", typ, err)
	}
	return ev
}

func jsonEqual(a, b []byte) bool {
	if isEmptyObject(a) && isEmptyObject(b) {
		return true
	}
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	ax, _ := json.Marshal(x)
	by, _ := json.Marshal(y)
	return bytes.Equal(ax, by)
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func intptr(n int) *int { return &n }
