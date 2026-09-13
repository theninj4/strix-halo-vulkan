package bench

import (
	"math"
	"testing"
)

// The mixed-width plan is the one part of IDEAS §1.12 that is pure host
// arithmetic, and it is the part a GPU run cannot check cheaply: a table whose
// groups are the wrong size, or whose output rows collide, produces numbers
// that look plausible. verifyMoEGEMVMixed catches that on the device by
// differencing against the width-1 build; these check the structure directly,
// including the invariant the buffer sizing depends on.

func testWidths() []int { return []int{1, 2, 4, 8} }

// walkPlan decomposes one count the way buildMoEGroupsMixed does, returning
// the widths in order.
func walkPlan(choice []int, count int) []int {
	var out []int
	for count > 0 {
		w := choice[count]
		out = append(out, w)
		if w >= count {
			break
		}
		count -= w
	}
	return out
}

func TestPlanMoEWidthsPrefersOneGroupWhenGroupsAreCheap(t *testing.T) {
	// A group worth 45 pad slots is gate_up's number at t=4: over-covering is
	// always cheaper than a second read of the expert.
	choice := planMoEWidths(12, testWidths(), moeGEMVCost{group: 45, slot: 1})
	for count, want := range map[int][]int{
		1: {1}, 3: {4}, 5: {8}, 8: {8}, 9: {8, 1}, 12: {8, 4},
	} {
		if got := walkPlan(choice, count); !equalInts(got, want) {
			t.Errorf("count %d: planned %v, want %v", count, got, want)
		}
	}
}

func TestPlanMoEWidthsPrefersExactCoverWhenSlotsAreExpensive(t *testing.T) {
	// A group cheaper than a slot is down's number at a serving batch, where
	// the fit puts one expert's weight read below the per-output-row cost of a
	// single activation row. Padding then never pays, so every plan covers its
	// count exactly.
	choice := planMoEWidths(12, testWidths(), moeGEMVCost{group: 1, slot: 45})
	for count := 1; count <= 12; count++ {
		plan := walkPlan(choice, count)
		total := 0
		for _, w := range plan {
			total += w
		}
		if total != count {
			t.Errorf("count %d: plan %v covers %d slots, want an exact cover", count, plan, total)
		}
	}
}

func TestPlanMoEWidthsPadsOnlyTheLastGroup(t *testing.T) {
	// The invariant every caller's buffer sizing rests on: only the last group
	// of an expert can be short, so an expert's pad is under max(widths).
	for _, c := range []moeGEMVCost{{group: 45, slot: 1}, {group: 6.5, slot: 1}, {group: 1, slot: 1}, {group: 1, slot: 45}} {
		choice := planMoEWidths(40, testWidths(), c)
		for count := 1; count <= 40; count++ {
			plan := walkPlan(choice, count)
			covered := 0
			for i, w := range plan {
				covered += w
				if covered > count && i != len(plan)-1 {
					t.Fatalf("cost %v count %d: plan %v pads group %d, not the last", c, count, plan, i)
				}
			}
			if pad := covered - count; pad < 0 || pad >= 8 {
				t.Fatalf("cost %v count %d: plan %v pads %d slots", c, count, plan, pad)
			}
		}
	}
}

func TestBuildMoEGroupsMixedLayout(t *testing.T) {
	const N = 64
	r := routeTokens(16)
	choice := planMoEWidths(r.maxRows, testWidths(), moeGEMVCost{group: 6.5, slot: 1})
	g := buildMoEGroupsMixed(r, N, choice, false)

	if g.pairs != r.rows() {
		t.Fatalf("plan covers %d pairs, routing has %d", g.pairs, r.rows())
	}
	if len(g.table) != 4*g.slots {
		t.Fatalf("table is %d words for %d slots", len(g.table), g.slots)
	}

	// Each part is a contiguous run of equal-width groups, and the parts tile
	// the table in order.
	slot := 0
	groups := 0
	for i, p := range g.parts {
		if p.firstSlot != slot {
			t.Fatalf("part %d starts at slot %d, want %d", i, p.firstSlot, slot)
		}
		if i > 0 && p.width >= g.parts[i-1].width {
			t.Fatalf("parts are not in descending width order: %v", g.parts)
		}
		slot += p.width * p.groups
		groups += p.groups
	}
	if slot != g.slots || groups != g.groups {
		t.Fatalf("parts cover %d slots / %d groups, table has %d / %d", slot, groups, g.slots, g.groups)
	}

	// The pair-level invariants: one slot per routed pair, at the output row
	// the width-1 table would have given it, every group on one expert, and
	// every pad at the end of its group writing the scratch row.
	one := buildMoEGroups(r, N, 1)
	want := map[[2]uint32]uint32{} // (bank row, token) -> output row
	for i := 0; i < one.slots; i++ {
		e := one.table[4*i]
		want[[2]uint32{e, one.table[4*i+1]}] = one.table[4*i+2]
	}
	scratch := uint32(r.rows())
	seen := map[uint32]bool{}
	for _, p := range g.parts {
		for gi := 0; gi < p.groups; gi++ {
			base := p.firstSlot + gi*p.width
			expert := g.table[4*base]
			pad := false
			for s := 0; s < p.width; s++ {
				row, tok, out := g.table[4*(base+s)], g.table[4*(base+s)+1], g.table[4*(base+s)+2]
				if row != expert {
					t.Fatalf("slot %d of a group names bank row %d, the group's first names %d", s, row, expert)
				}
				if out == scratch {
					pad = true
					continue
				}
				if pad {
					t.Fatalf("group at slot %d has a real slot after a pad", base)
				}
				if seen[out] {
					t.Fatalf("output row %d claimed twice", out)
				}
				seen[out] = true
				if w, ok := want[[2]uint32{row, tok}]; !ok || w != out {
					t.Fatalf("(bank row %d, token %d) got output row %d, the width-1 table says %d (present=%v)",
						row, tok, out, w, ok)
				}
			}
		}
	}
	if len(seen) != g.pairs {
		t.Fatalf("%d output rows written, %d pairs routed", len(seen), g.pairs)
	}
}

func TestBuildMoEGroupsMixedWholeKeepsAnExpertInOnePart(t *testing.T) {
	// The `whole` plan's defining property: every group of an expert has the
	// same width, so they all land in one part and its weights are read
	// back-to-back rather than in two dispatches.
	const N = 64
	r := routeTokens(64)
	choice := planMoEWidthWhole(r.maxRows, testWidths(), moeGEMVCost{group: 2, slot: 1})
	g := buildMoEGroupsMixed(r, N, choice, true)
	if g.pairs != r.rows() {
		t.Fatalf("plan covers %d pairs, routing has %d", g.pairs, r.rows())
	}
	partOf := map[uint32]int{}
	for i, p := range g.parts {
		for gi := 0; gi < p.groups; gi++ {
			expert := g.table[4*(p.firstSlot+gi*p.width)]
			if prev, seen := partOf[expert]; seen && prev != i {
				t.Fatalf("expert at bank row %d appears in parts %d and %d", expert, prev, i)
			}
			partOf[expert] = i
		}
	}
	if len(partOf) != r.touched {
		t.Fatalf("%d experts in the table, routing touched %d", len(partOf), r.touched)
	}
	if g.crossPart != 0 {
		t.Fatalf("a whole-expert plan reports %d groups in a second part", g.crossPart)
	}
}

func TestBuildMoEGroupsMixedSplitCountsItsCrossings(t *testing.T) {
	// The split plan's crossPart is what the whole plan's zero is measured
	// against, so it has to be the real count and not a proxy: recount it from
	// the table.
	const N = 64
	r := routeTokens(64)
	choice := planMoEWidths(r.maxRows, testWidths(), moeGEMVCost{group: 1, slot: 45})
	g := buildMoEGroupsMixed(r, N, choice, false)
	parts := map[uint32]map[int]bool{}
	for i, p := range g.parts {
		for gi := 0; gi < p.groups; gi++ {
			e := g.table[4*(p.firstSlot+gi*p.width)]
			if parts[e] == nil {
				parts[e] = map[int]bool{}
			}
			parts[e][i] = true
		}
	}
	want := 0
	for _, ps := range parts {
		want += len(ps) - 1
	}
	if g.crossPart != want {
		t.Fatalf("crossPart is %d, the table says %d", g.crossPart, want)
	}
	if want == 0 {
		t.Fatal("this routing did not make the split plan cross a part at all; the test proves nothing")
	}
}

func TestFitMoEGEMVCostRecoversItsOwnCoefficients(t *testing.T) {
	// Synthetic rows in the shape the M sweep produces, with a known (group,
	// slot) behind them: the fit has to come back with it.
	const layer, tokens, nrows = "moe.down", 256, 4
	want := moeGEMVCost{group: 613.6, slot: 680.4}
	var results []Result
	for _, v := range moeGEMVVariants {
		if v.mixed || v.strideArm || v.rowsPerWave() != nrows || v.wavesPerWG() != 1 {
			continue
		}
		if v.weightsPerLoad != 32 || v.block != 128 || v.waveSize != 0 {
			continue
		}
		groups, slots := 2560/v.mblock(), 2560
		results = append(results, Result{
			Op: "moe", Variant: v.name, WeightFormat: "w4a8",
			NsPerIter: want.predict(groups, slots),
			Detail: "layer=" + layer + ";mode=" + mode1 + ";tokens=256;groups=" +
				itoa(groups) + ";slots=" + itoa(slots),
		})
	}
	if len(results) < 2 {
		t.Fatalf("only %d synthetic rows; the M sweep at NROWS=%d should have more", len(results), nrows)
	}
	got, ok := fitMoEGEMVCost(results, layer, tokens, nrows, 32, 128, 0)
	if !ok {
		t.Fatal("no fit")
	}
	if math.Abs(got.group-want.group) > 1e-6*want.group || math.Abs(got.slot-want.slot) > 1e-6*want.slot {
		t.Fatalf("fitted %+v, want %+v", got, want)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
