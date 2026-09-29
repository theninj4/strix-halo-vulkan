package dit

import "testing"

// TestCoopMatLayout pins the element order h3_attn_t.comp is written
// against (VIDEO.md M11c; checkCoopMatLayout says which). On a device where
// it fails, the build runs the plain attention instead.
func TestCoopMatLayout(t *testing.T) {
	dev, done := newTestDevice(t)
	defer done()
	lay, err := coopMatLayout(dev)
	if err != nil {
		t.Fatal(err)
	}
	for u, name := range []string{"A", "B", "acc32", "acc16"} {
		for _, l := range []int{0, 1, 16, 17} {
			t.Logf("%s lane %2d: %v", name, l, lay[u][l])
		}
	}
	if err := checkCoopMatLayout(dev); err != nil {
		t.Fatal(err)
	}
}
