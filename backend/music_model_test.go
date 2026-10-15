package backend

import "testing"

// TestMusicResolveModel: a request's model picks a staged DiT by id, short
// name or a word in it, and anything else runs the default.
func TestMusicResolveModel(t *testing.T) {
	both := &Music{id: "acestep-v15-xl-base", dits: []string{"acestep-v15-xl-base", "acestep-v15-xl-turbo"}}
	baseOnly := &Music{id: "acestep-v15-xl-base", dits: []string{"acestep-v15-xl-base"}}
	turboFirst := &Music{id: "acestep-v15-xl-turbo", dits: []string{"acestep-v15-xl-turbo", "acestep-v15-xl-base"}}
	for _, c := range []struct {
		b     *Music
		model string
		want  string
	}{
		{both, "", "acestep-v15-xl-base"},
		{both, "acestep-v15-xl-turbo", "acestep-v15-xl-turbo"},
		{both, "turbo", "acestep-v15-xl-turbo"},
		{both, "TURBO", "acestep-v15-xl-turbo"},
		{both, "acestep-v15-turbo", "acestep-v15-xl-turbo"}, // upstream's 2B default runs XL turbo
		{both, "acestep-v15-xl-sft", "acestep-v15-xl-base"}, // not staged: the default
		{both, "gpt-4o", "acestep-v15-xl-base"},
		{both, "turbocharged", "acestep-v15-xl-base"}, // a word, not a substring
		{baseOnly, "turbo", "acestep-v15-xl-base"},
		{turboFirst, "", "acestep-v15-xl-turbo"},
		{turboFirst, "base", "acestep-v15-xl-base"},
	} {
		if got := c.b.resolveModel(c.model); got != c.want {
			t.Errorf("dits %v, model %q: %s, want %s", c.b.dits, c.model, got, c.want)
		}
	}
}
