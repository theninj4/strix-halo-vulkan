package api

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// rawRequest is jsonRequest for a body written out by hand, which is how the
// envelope's own decoding -- a string where an array is also allowed, an
// unknown field -- gets exercised rather than round-tripped through the Go
// type.
func rawRequest(method, path, body string) *http.Request {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

// fakeEmbedding is the third fake: no checkpoint, no device, and it records
// what the routing layer handed it, which is the half of this endpoint worth
// testing here.
type fakeEmbedding struct {
	req  *EmbeddingRequest
	err  error
	vecs [][]float32
}

func (f *fakeEmbedding) Models() []Model {
	return []Model{{ID: "qwen3-embedding-0.6b", Object: "model", OwnedBy: "local"}}
}

func (f *fakeEmbedding) Embed(_ context.Context, req *EmbeddingRequest) (*EmbeddingResult, error) {
	f.req = req
	if f.err != nil {
		return nil, f.err
	}
	vecs := f.vecs
	if vecs == nil {
		vecs = make([][]float32, len(req.Input))
		for i := range vecs {
			vecs[i] = []float32{float32(i), 0.5, -0.5, 1}
		}
	}
	return &EmbeddingResult{Vectors: vecs, Usage: Usage{PromptTokens: 7, TotalTokens: 7}}, nil
}

func TestEmbeddings(t *testing.T) {
	f := &fakeEmbedding{}
	s := &Server{Embedding: f}
	rec := do(t, s, jsonRequest("POST", "/v1/embeddings", EmbeddingRequest{
		Input: StringList{"one", "two"}, Model: "text-embedding-3-small",
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var resp EmbeddingResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "list" || len(resp.Data) != 2 {
		t.Fatalf("object %q, %d items", resp.Object, len(resp.Data))
	}
	// The response reports what ran, not what was asked for, as every other
	// endpoint here does.
	if resp.Model != "qwen3-embedding-0.6b" {
		t.Errorf("model %q", resp.Model)
	}
	for i, item := range resp.Data {
		if item.Object != "embedding" || item.Index != i {
			t.Errorf("item %d: object %q index %d", i, item.Object, item.Index)
		}
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 7 || resp.Usage.TotalTokens != 7 {
		t.Errorf("usage %+v", resp.Usage)
	}
}

// A single string and an array of strings are the same request, which is the
// one thing about OpenAI's envelope a client is most likely to exercise both
// ways.
func TestEmbeddingsInputForms(t *testing.T) {
	for _, body := range []string{
		`{"input": "one"}`,
		`{"input": ["one"]}`,
	} {
		f := &fakeEmbedding{}
		rec := do(t, &Server{Embedding: f}, rawRequest("POST", "/v1/embeddings", body))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d: %s", body, rec.Code, rec.Body)
		}
		if len(f.req.Input) != 1 || f.req.Input[0] != "one" {
			t.Errorf("%s: backend saw %v", body, f.req.Input)
		}
	}
}

// base64 is OpenAI's other encoding: little-endian float32. A client that
// asked for it and got JSON numbers would decode garbage, so the bytes are
// checked rather than the field's type.
func TestEmbeddingsBase64(t *testing.T) {
	f := &fakeEmbedding{vecs: [][]float32{{1, -2, 0.25}}}
	rec := do(t, &Server{Embedding: f}, jsonRequest("POST", "/v1/embeddings", EmbeddingRequest{
		Input: StringList{"one"}, EncodingFormat: "base64",
	}))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var resp EmbeddingResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	enc, ok := resp.Data[0].Embedding.(string)
	if !ok {
		t.Fatalf("embedding is %T, want a base64 string", resp.Data[0].Embedding)
	}
	raw, err := base64.StdEncoding.DecodeString(enc)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 12 {
		t.Fatalf("%d bytes for 3 float32", len(raw))
	}
	for i, want := range []float32{1, -2, 0.25} {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		if got != want {
			t.Errorf("component %d is %g, want %g", i, got, want)
		}
	}
}

// The instruction is this API's one extension, and it has to reach the
// backend untouched: it is the difference between a query vector and a
// document vector.
func TestEmbeddingsInstruct(t *testing.T) {
	f := &fakeEmbedding{}
	rec := do(t, &Server{Embedding: f}, rawRequest("POST", "/v1/embeddings",
		`{"input": "one", "instruct": "Retrieve documents about cats", "dimensions": 32}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if f.req.Instruct != "Retrieve documents about cats" || f.req.Dimensions != 32 {
		t.Errorf("backend saw instruct %q, dimensions %d", f.req.Instruct, f.req.Dimensions)
	}
}

func TestEmbeddingsRejects(t *testing.T) {
	for _, c := range []struct{ name, body string }{
		{"no input", `{}`},
		{"empty string", `{"input": ""}`},
		{"empty in array", `{"input": ["one", ""]}`},
		{"bad encoding", `{"input": "one", "encoding_format": "float16"}`},
		{"negative dimensions", `{"input": "one", "dimensions": -1}`},
	} {
		f := &fakeEmbedding{}
		rec := do(t, &Server{Embedding: f}, rawRequest("POST", "/v1/embeddings", c.body))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d: %s", c.name, rec.Code, rec.Body)
		}
		if f.req != nil {
			t.Errorf("%s: reached the backend", c.name)
		}
	}
}

// A backend that returned the wrong number of vectors would misalign a
// client's corpus silently, so it is a 500 and not a response.
func TestEmbeddingsVectorCountChecked(t *testing.T) {
	f := &fakeEmbedding{vecs: [][]float32{{1, 2, 3}}}
	rec := do(t, &Server{Embedding: f}, jsonRequest("POST", "/v1/embeddings", EmbeddingRequest{
		Input: StringList{"one", "two"},
	}))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

func TestEmbeddingsNotLoaded(t *testing.T) {
	rec := do(t, &Server{}, jsonRequest("POST", "/v1/embeddings", EmbeddingRequest{
		Input: StringList{"one"},
	}))
	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), "-embed") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

// And the model list, which is how a client discovers this instance embeds.
func TestEmbeddingsListed(t *testing.T) {
	rec := do(t, &Server{Embedding: &fakeEmbedding{}}, httptest.NewRequest("GET", "/v1/models", http.NoBody))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "qwen3-embedding-0.6b") {
		t.Errorf("model list does not name the embedding model: %s", rec.Body)
	}
}
