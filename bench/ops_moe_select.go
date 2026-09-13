package bench

import (
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"strix-halo-vulkan/vk"
)

// This file is IDEAS §1.12: the M block sized from the routing histogram
// instead of compiled in.
//
// Where it comes from. §1.11 left one cell of the decode path under the bus —
// `down` at 256 sequences in flight, reading 66% of DRAM where every other
// cell reads 93-100% — and priced about a quarter of that gap as **pad slots**.
// The M block is a compile-time constant, so a build with MROWS=4 covers an
// expert that routing gave 5 pairs with two groups of four and three padding
// slots; at t=256 its best build runs 829 groups against 2560 routed pairs,
// so 756 of 3316 slots (23%) are doing a real slot's work and writing an
// output row nothing reads. Routing does not hand out multiples of four.
//
// What is missing is not another binary. §1.11 finding 3 already fitted the
// cost of the two things a table is made of — a **group**, which is one read
// of an expert's 845 KB, and a **slot**, which is one activation row, one
// wave's share of the launch and (§1.10) a fixed per-output-row cost — and an
// engine that knows both numbers and the histogram can *choose*. Two choices,
// and this file measures them apart:
//
//  1. **Which fixed build to dispatch.** Given the counts, predict the cost of
//     every built width and take the cheapest. It costs nothing at run time
//     (the histogram is already in the routing table) and it is only as good
//     as the cost model, so what it needs is the oracle beside it: the width
//     that actually won, measured.
//
//  2. **Several widths at once.** A group's slots must all name one expert,
//     but nothing says two groups must be the same size — the width is fixed
//     per *dispatch*, not per table. So an expert with 5 routed pairs can be
//     covered by one group of 8 (3 pads, one weight read) or by a 4 and a 1
//     (no pads, two weight reads), and which is cheaper is exactly the ratio
//     the cost model carries. Decompose every expert that way, bucket the
//     groups by width, and dispatch each bucket with the build for that width.
//     The buckets write disjoint output rows, so they need no barrier between
//     them and the kernel does not change at all.
//
// The second is the one that can remove the pad rather than trade it, and the
// cost model is what picks the decomposition, so the two halves are the same
// experiment: a plan is only as good as the numbers behind it, and the fit is
// printed next to what it bought.

// moeGEMVCost is the two-term cost model IDEAS §1.11 finding 3 fitted: a
// dispatch's time is its groups times what a group costs plus its slots times
// what a slot costs. Both are in nanoseconds and both are per *dispatch*, so
// they absorb everything that scales with the two counts — a group's weight
// read and whatever of it the MALL serves, a slot's activation row, its
// output rows and its share of the launch.
//
// It is a fit and not a derivation, and the two coefficients are not stable
// across batches: at t=4 every expert has one pair and the whole bank is
// cache-resident, at t=256 five pairs share an expert and it is not. Which is
// why the summary prints the model calibrated at the batch *and* calibrated
// once at t=4, next to the width that actually won.
type moeGEMVCost struct {
	group float64 // ns per group (one expert's weights)
	slot  float64 // ns per slot (one activation row through all its output rows)
}

func (c moeGEMVCost) ok() bool { return c.group > 0 && c.slot > 0 }

func (c moeGEMVCost) predict(groups, slots int) float64 {
	return c.group*float64(groups) + c.slot*float64(slots)
}

// ratio is how many pad slots a group is worth — §1.11 finding 3's engine
// number (~45 on gate_up, ~6.5 on down at t=4) and the thing that decides
// whether an under-covered expert should be given a bigger group or a second
// one.
func (c moeGEMVCost) ratio() float64 {
	if c.slot <= 0 {
		return 0
	}
	return c.group / c.slot
}

// fitMoEGEMVCost fits (group, slot) from the fixed-width rows already
// measured for one (layer, batch, NROWS) cell: the M sweep is four dispatches
// whose (groups, slots) pairs differ by construction, so two coefficients are
// over-determined by two points and fitted by least squares over all of them.
//
// It is a non-negative fit. An unconstrained solve can put one coefficient
// below zero when the four points nearly line up — down's t=256 cell has a
// group costing less than a slot, which is physically fine, but a *negative*
// group would make the planner buy groups for free — so a solve that leaves
// the feasible region is replaced by the better of the two one-term fits.
func fitMoEGEMVCost(results []Result, layer string, tokens, nrows, weightsPerLoad, block int, wave uint32) (moeGEMVCost, bool) {
	type point struct{ groups, slots, ns float64 }
	var pts []point
	for _, r := range results {
		if r.WeightFormat != "w4a8" || detailField(r.Detail, "mode") != mode1 {
			continue
		}
		if detailField(r.Detail, "layer") != layer || atoiOr(detailField(r.Detail, "tokens")) != tokens {
			continue
		}
		v, ok := lookupMoEGEMVVariant(r.Variant)
		if !ok || v.mixed || v.strideArm || v.wavesPerWG() != 1 {
			continue
		}
		if v.rowsPerWave() != nrows || v.weightsPerLoad != weightsPerLoad || v.block != block || v.waveSize != wave {
			continue
		}
		pts = append(pts, point{
			groups: float64(atoiOr(detailField(r.Detail, "groups"))),
			slots:  float64(atoiOr(detailField(r.Detail, "slots"))),
			ns:     r.NsPerIter,
		})
	}
	if len(pts) < 2 {
		return moeGEMVCost{}, false
	}

	var gg, gs, ss, gt, st float64
	for _, p := range pts {
		gg += p.groups * p.groups
		gs += p.groups * p.slots
		ss += p.slots * p.slots
		gt += p.groups * p.ns
		st += p.slots * p.ns
	}
	sse := func(c moeGEMVCost) float64 {
		var e float64
		for _, p := range pts {
			d := c.group*p.groups + c.slot*p.slots - p.ns
			e += d * d
		}
		return e
	}
	if det := gg*ss - gs*gs; math.Abs(det) > 0 {
		c := moeGEMVCost{group: (gt*ss - st*gs) / det, slot: (gg*st - gs*gt) / det}
		if c.ok() {
			return c, true
		}
	}
	// Both one-term fits, since the joint one left the feasible region.
	var best moeGEMVCost
	bestSSE := math.Inf(1)
	if gg > 0 {
		if c := (moeGEMVCost{group: gt / gg, slot: 0}); c.group > 0 && sse(c) < bestSSE {
			best, bestSSE = c, sse(c)
		}
	}
	if ss > 0 {
		if c := (moeGEMVCost{group: 0, slot: st / ss}); c.slot > 0 && sse(c) < bestSSE {
			best, bestSSE = c, sse(c)
		}
	}
	if math.IsInf(bestSSE, 1) {
		return moeGEMVCost{}, false
	}
	// A zero coefficient is a usable predictor but not a usable planner (it
	// makes one of the two things free), so nudge it to a floor that keeps the
	// decomposition finite without changing which build the rule picks.
	if best.group <= 0 {
		best.group = best.slot / 1e3
	}
	if best.slot <= 0 {
		best.slot = best.group / 1e3
	}
	return best, true
}

// planMoEWidths solves, for every routed count up to maxCount, the cheapest
// way to cover it with the widths that are built: choice[c] is the width of
// the *first* group to cut off an expert with c pairs left, and the rest
// follows by recursion.
//
// The recursion is the whole engine rule in three lines. Covering c pairs with
// a group of width m either finishes the expert (m >= c, paying m-c pad slots)
// or takes m of them and leaves c-m, and a group costs what a group costs
// either way. At a large group-to-slot ratio (gate_up, where a group is ~45
// slots) it always pays to over-cover; at a small one (down at a serving
// batch, where a group is worth about one slot) it never does. Nothing in
// between is guessable, which is why it is solved rather than stated.
//
// Only the last group of an expert can be padded — every earlier step takes a
// full m — so an expert's pad is at most max(widths)-1 and the table the
// caller allocates for a fixed block of that width still fits.
func planMoEWidths(maxCount int, widths []int, c moeGEMVCost) []int {
	// Widest first, so that a tie — which is what a group costing exactly one
	// slot is — is broken towards the bigger first group rather than towards a
	// trail of small ones.
	sorted := append([]int(nil), widths...)
	sort.Sort(sort.Reverse(sort.IntSlice(sorted)))
	choice := make([]int, maxCount+1)
	cost := make([]float64, maxCount+1)
	for n := 1; n <= maxCount; n++ {
		cost[n] = math.Inf(1)
		for _, m := range sorted {
			// A group of width m costs one group and m slots whether or not
			// all m of them are real — which is the whole reason an
			// over-covered expert is ever worth it, and the reason the real
			// slots are charged here as well as the pads: their total is the
			// same in every plan, but leaving them out of one branch and not
			// the other would not be.
			t := c.group + c.slot*float64(m)
			if m < n {
				t += cost[n-m]
			}
			if t < cost[n] {
				cost[n], choice[n] = t, m
			}
		}
		if math.IsInf(cost[n], 1) {
			return nil
		}
	}
	return choice
}

// planMoEWidthWhole is the other way to use the same freedom, and the one that
// keeps an expert inside a single dispatch: pick *one* width for the expert
// and cover its c pairs with ceil(c/w) groups of it, padding only the last.
//
// The difference from planMoEWidths is what it gives up and what it buys. It
// cannot spend a 4 and a 1 on an expert routed five pairs, so it pads more;
// but every group of that expert is then in one part, running back-to-back
// over weights the MALL has just fetched, where a decomposition puts the 4 and
// the 1 in dispatches separated by all the other experts' traffic. The M block
// is a cache lever (§1.9 finding 2), so that separation is not free, and these
// two plans are what price it against each other.
func planMoEWidthWhole(maxCount int, widths []int, c moeGEMVCost) []int {
	sorted := append([]int(nil), widths...)
	sort.Sort(sort.Reverse(sort.IntSlice(sorted)))
	choice := make([]int, maxCount+1)
	for n := 1; n <= maxCount; n++ {
		best := math.Inf(1)
		for _, m := range sorted {
			groups := (n + m - 1) / m
			if t := float64(groups) * (c.group + c.slot*float64(m)); t < best {
				best, choice[n] = t, m
			}
		}
	}
	return choice
}

// moeMixedPart is one dispatch of a mixed-width plan: every group in it is
// `width` slots, they are contiguous in the table from `firstSlot`, and the
// build compiled for that width covers them.
type moeMixedPart struct {
	width     int
	firstSlot int
	groups    int
}

// moeMixedGroups is the mixed-width table: the same four-word slot entries
// buildMoEGroups writes, ordered so that each width owns one contiguous run,
// plus the list of dispatches that reads them.
//
// The output rows are numbered exactly as the fixed-width table numbers them —
// expert-major, in routing order — so a mixed dispatch and any single-width
// build leave *the same* y buffer, which is what lets the mixed path be
// checked against a build that is already verified instead of against a second
// copy of the reference.
type moeMixedGroups struct {
	table  []uint32
	parts  []moeMixedPart
	pairs  int
	slots  int
	groups int
	// crossPart is how many of an expert's groups land in a *different*
	// dispatch from its first — the thing the two plans differ in, and the
	// one quantity the cost model cannot see. A `whole` plan gives an expert
	// one width, so every group of it is in one part and this is zero; a
	// `split` plan spends several widths on one expert and each is a separate
	// dispatch, so its weights are fetched again after everything else in the
	// part between them has run.
	crossPart int
}

// widthCounts renders the plan the way the summary reports it: how many
// groups each width took, widest first.
func (m moeMixedGroups) widthCounts() string {
	var out []string
	for _, p := range m.parts {
		out = append(out, fmt.Sprintf("%dx%d", p.width, p.groups))
	}
	return strings.Join(out, "/")
}

// buildMoEGroupsMixed lays the plan out. `whole` says the choice table names
// one width for the whole expert (planMoEWidthWhole) rather than the width of
// its next group (planMoEWidths), which is the difference between an expert
// living in one dispatch and living in several.
func buildMoEGroupsMixed(r moeRouting, N int, choice []int, whole bool) moeMixedGroups {
	// An expert's decomposition, in the order its groups will run.
	type cut struct {
		expert int
		first  int // index into r.perExpert[expert]
		take   int // real pairs in this group; the rest of `width` is pad
	}
	byWidth := map[int][]cut{}
	var widths []int
	// The pair index of expert e's first token, which is its first output row:
	// the same numbering buildMoEGroups gives, since both walk experts in
	// order.
	pairBase := make([]int, moeExperts)
	pairs := 0
	for e := 0; e < moeExperts; e++ {
		pairBase[e] = pairs
		pairs += len(r.perExpert[e])
	}
	for e := 0; e < moeExperts; e++ {
		left, pos := len(r.perExpert[e]), 0
		w := choice[left]
		for left > 0 {
			if !whole {
				w = choice[left]
			}
			take := w
			if take > left {
				take = left
			}
			if _, seen := byWidth[w]; !seen {
				widths = append(widths, w)
			}
			byWidth[w] = append(byWidth[w], cut{e, pos, take})
			pos += take
			left -= take
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(widths)))

	g := moeMixedGroups{pairs: pairs}
	// One part per width, so an expert crosses a part boundary once per
	// distinct width it was given beyond the first.
	perExpertWidths := map[int]map[int]bool{}
	for w, cuts := range byWidth {
		for _, c := range cuts {
			if perExpertWidths[c.expert] == nil {
				perExpertWidths[c.expert] = map[int]bool{}
			}
			perExpertWidths[c.expert][w] = true
		}
	}
	for _, ws := range perExpertWidths {
		g.crossPart += len(ws) - 1
	}
	scratch := uint32(r.rows())
	for _, w := range widths {
		cuts := byWidth[w]
		g.parts = append(g.parts, moeMixedPart{width: w, firstSlot: g.slots, groups: len(cuts)})
		for _, c := range cuts {
			toks := r.perExpert[c.expert]
			for i := 0; i < w; i++ {
				if i < c.take {
					t := toks[c.first+i]
					g.table = append(g.table, uint32(c.expert*N), uint32(t), uint32(pairBase[c.expert]+c.first+i), 0)
				} else {
					// The same pad a short fixed-width group takes: the
					// neighbouring (cache-hot) activation row, and an output
					// row past every real one.
					g.table = append(g.table, uint32(c.expert*N), uint32(toks[c.first+c.take-1]), scratch, 0)
				}
				g.slots++
			}
			g.groups++
		}
	}
	return g
}

// moeSelectBarrierArms is what separates the parts of a mixed dispatch: a real
// engine knows they write disjoint output rows and lets them overlap, and the
// second arm is what the split costs if it cannot (§1.8 finding 3 measured a
// barrier at 690-1555 ns against a memory-bound dispatch, so over three or
// four parts this is a prediction with a number behind it).
var moeSelectBarrierArms = []struct {
	mode     string
	barriers bool
}{
	// The overlapping arm is recorded under the plain grouped mode, because
	// that is what it is — a grouped dispatch of the same kernels over the
	// same table — and because the summaries that pick the fastest decode
	// kernel at a batch should be choosing between it and the fixed builds.
	// Its `widths` and `parts` fields are what tell it apart in the CSV.
	{mode: mode1, barriers: false},
	{mode: "mixed_bar", barriers: true},
}

// runMoEGEMVSelection runs the mixed-width plans over the bank the fixed
// builds were just measured on. It runs last because it plans with a cost
// model fitted to those measurements: the planner is being given the best
// numbers this machine can supply, so that what the arm measures is whether
// the *decomposition* is worth anything, not whether a stale calibration is.
// (What a stale calibration would have picked is in the summary, from the same
// rows, at no GPU cost.)
func runMoEGEMVSelection(dev *vk.Device, s moeShape, variants []moeGEMVVariant, pipes map[string]*vk.ComputePipeline,
	routings map[int]moeRouting, tokens []int, measured []Result, bufs gemvBuffers, warmup, iters uint32) ([]Result, error) {
	var out []Result
	for _, v := range variants {
		if !v.mixed {
			continue
		}
		ldw, ok := v.ldw(s.K)
		if !ok || s.N%v.rowsPerWG() != 0 {
			continue
		}
		// The builds this plan can dispatch: same width, wave and
		// quantization block, same row block, one wave per workgroup, every
		// M width that exists.
		byWidth := map[int]moeGEMVVariant{}
		var widths []int
		for _, c := range moeGEMVVariants {
			if c.mixed || c.strideArm || c.wavesPerWG() != 1 || c.rowsPerWave() != v.rowsPerWave() {
				continue
			}
			if c.weightsPerLoad != v.weightsPerLoad || c.block != v.block || c.waveSize != v.waveSize {
				continue
			}
			if pipes[c.name] == nil {
				continue
			}
			byWidth[c.mblock()] = c
			widths = append(widths, c.mblock())
		}
		if len(widths) < 2 {
			fmt.Fprintf(os.Stderr, "moe gemv %s %s: only %d M widths built, skipping\n", s.layer, v.name, len(widths))
			continue
		}
		sort.Ints(widths)

		for _, t := range tokens {
			r := routings[t]
			cost, fitted := fitMoEGEMVCost(measured, s.layer, t, v.rowsPerWave(), v.weightsPerLoad, v.block, v.waveSize)
			if !fitted {
				fmt.Fprintf(os.Stderr, "moe gemv %s %s t=%d: no cost fit, skipping\n", s.layer, v.name, t)
				continue
			}
			choice := moeSelectPlan(v, r.maxRows, widths, cost)
			if choice == nil {
				continue
			}
			g := buildMoEGroupsMixed(r, s.N, choice, v.wholeExpert)
			if g.pairs != r.rows() {
				return nil, fmt.Errorf("moe gemv %s %s t=%d: plan covers %d of %d pairs",
					s.layer, v.name, t, g.pairs, r.rows())
			}
			if got, want := len(g.table)*4, bufs.groups.Size(); got > want {
				return nil, fmt.Errorf("moe gemv %s %s t=%d: %d-slot table needs %d B of a %d B buffer",
					s.layer, v.name, t, g.slots, got, want)
			}

			if err := verifyMoEGEMVMixed(s, v, byWidth, pipes, bufs, r, g, ldw); err != nil {
				return nil, fmt.Errorf("moe gemv %s %s t=%d correctness check: %w", s.layer, v.name, t, err)
			}

			dispatches := make([]vk.MultiDispatch, 0, len(g.parts))
			for _, p := range g.parts {
				dispatches = append(dispatches, vk.MultiDispatch{
					Pipeline:      pipes[byWidth[p.width].name],
					GroupsX:       uint32(s.N / v.rowsPerWG()),
					GroupsY:       uint32(p.groups),
					PushConstants: moeGEMVPushConstants(s.N, s.K, v.block, ldw, s.K, p.firstSlot, 1.0/127.0),
				})
			}

			for _, arm := range moeSelectBarrierArms {
				base := moeGEMVCounts(s, v, r, g.pairs, g.slots, g.groups, ldw)
				base.extra = fmt.Sprintf(";plan=%s;widths=%s;parts=%d;crosspart=%d;barriers=%d;groupus=%.4f;slotus=%.4f",
					moeSelectPlanName(v), g.widthCounts(), len(g.parts), g.crossPart, boolToInt(arm.barriers),
					cost.group/1e3, cost.slot/1e3)
				ns, clocks, err := TimeDispatchMulti(dispatches, 1, warmup, iters, arm.barriers)
				if err != nil {
					return nil, fmt.Errorf("moe gemv %s %s %s t=%d: %w", s.layer, v.name, arm.mode, t, err)
				}
				out = append(out, base.finish(arm.mode, len(g.parts), ns, clocks))
			}
		}
	}
	return out, nil
}

// moeSelectPlan is the plan a mixed variant asks for: a per-group
// decomposition, or one width for the whole expert.
func moeSelectPlan(v moeGEMVVariant, maxCount int, widths []int, c moeGEMVCost) []int {
	if v.wholeExpert {
		return planMoEWidthWhole(maxCount, widths, c)
	}
	return planMoEWidths(maxCount, widths, c)
}

func moeSelectPlanName(v moeGEMVVariant) string {
	if v.wholeExpert {
		return "whole"
	}
	return "split"
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// verifyMoEGEMVMixed checks the mixed-width dispatch against the single-width
// build of the same row block, on the real bank and the real routing.
//
// It is an equivalence check rather than a reference check, and deliberately:
// the reference at these shapes is 507 experts of a 2560x2560 matrix, and the
// thing that can go wrong here is not the arithmetic — every build in the plan
// is already checked against the CPU by verifyMoEGroupedGEMV — but the
// *addressing*: a part's groupBase, its grid extent, the output row a slot
// claims, and whether two parts write over each other. All four show up as a
// disagreement with the width-1 table, which numbers its output rows the same
// way and reduces each row in the same order, so the two runs agree to the
// bit when the plan is right.
func verifyMoEGEMVMixed(s moeShape, v moeGEMVVariant, byWidth map[int]moeGEMVVariant, pipes map[string]*vk.ComputePipeline,
	bufs gemvBuffers, r moeRouting, g moeMixedGroups, ldw int) error {
	plain, ok := byWidth[1]
	if !ok || pipes[plain.name] == nil {
		return fmt.Errorf("no width-1 build to check against")
	}
	rows := r.rows() * s.N

	one := buildMoEGroups(r, s.N, 1)
	bufs.groups.WriteBytes(uint32SliceToBytes(one.table))
	pc := moeGEMVPushConstants(s.N, s.K, v.block, ldw, s.K, 0, 1.0/127.0)
	if _, err := pipes[plain.name].DispatchTimed(uint32(s.N/v.rowsPerWG()), uint32(one.groups), 1, 1, pc); err != nil {
		return err
	}
	want := bufs.y.ReadFloat32(rows)

	bufs.groups.WriteBytes(uint32SliceToBytes(g.table))
	dispatches := make([]vk.MultiDispatch, 0, len(g.parts))
	for _, p := range g.parts {
		dispatches = append(dispatches, vk.MultiDispatch{
			Pipeline:      pipes[byWidth[p.width].name],
			GroupsX:       uint32(s.N / v.rowsPerWG()),
			GroupsY:       uint32(p.groups),
			PushConstants: moeGEMVPushConstants(s.N, s.K, v.block, ldw, s.K, p.firstSlot, 1.0/127.0),
		})
	}
	if _, err := vk.DispatchMultiTimed(dispatches, 1, 1, false); err != nil {
		return err
	}
	got := bufs.y.ReadFloat32(rows)

	for i := range want {
		if !closeEnough(got[i], want[i], 1e-4) {
			return fmt.Errorf("output row %d element %d: mixed %g, width-1 %g",
				i/s.N, i%s.N, got[i], want[i])
		}
	}
	return nil
}

// closeEnough is a relative comparison with an absolute floor, for values that
// should be bit-identical and are only being given room for a compiler that
// reassociates something.
func closeEnough(got, want float32, tol float64) bool {
	d := math.Abs(float64(got) - float64(want))
	return d <= tol || d <= tol*math.Abs(float64(want))
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

// moeSelectCalibrationTokens is the batch a "calibrate once, dispatch
// anywhere" engine would fit its cost model at. It is the smallest batch whose
// experts all have exactly one routed pair, which makes the fit a clean
// separation — the group count is the same at every M width, so the slope over
// the width is the slot and the intercept is the group — and it is the batch a
// model would be calibrated at if the calibration were cheap rather than
// representative.
const moeSelectCalibrationTokens = 4

// printMoEGEMVSelection is IDEAS §1.12's two questions, in two tables: whether
// a cost model picks the right fixed width off the histogram, and whether
// covering the histogram with several widths at once beats the best fixed one.
func printMoEGEMVSelection(w io.Writer, results []Result) {
	type key struct {
		layer  string
		tokens int
		nrows  int
	}
	type row struct {
		variant              string
		ns                   float64
		groups, slots, pad   int
		mrows                int
		pairs, experts       int
		parts                int
		crossPart            int
		widths               string
		groupUs, slotUs      float64
		mixedNs, mixedBarNs  float64
		mixedGroups, mixedSl int
		mixedParts           int
	}
	type planKey struct {
		key
		plan string
	}
	fixed := map[key][]row{}
	mixed := map[planKey]*row{}
	plans := map[key][]string{}
	var keys []key
	seen := map[key]bool{}
	note := func(k key) {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	for _, r := range results {
		if r.WeightFormat != "w4a8" {
			continue
		}
		v, ok := lookupMoEGEMVVariant(r.Variant)
		if !ok || v.strideArm || v.wavesPerWG() != 1 {
			continue
		}
		k := key{detailField(r.Detail, "layer"), atoiOr(detailField(r.Detail, "tokens")), v.rowsPerWave()}
		mode := detailField(r.Detail, "mode")
		cur := row{
			variant: r.Variant, ns: r.NsPerIter, mrows: v.mblock(),
			groups:  atoiOr(detailField(r.Detail, "groups")),
			slots:   atoiOr(detailField(r.Detail, "slots")),
			pad:     atoiOr(detailField(r.Detail, "padslots")),
			pairs:   atoiOr(detailField(r.Detail, "pairs")),
			experts: atoiOr(detailField(r.Detail, "experts")),
		}
		switch {
		case v.mixed:
			if mode != mode1 && mode != "mixed_bar" {
				continue
			}
			note(k)
			pk := planKey{k, detailField(r.Detail, "plan")}
			m := mixed[pk]
			if m == nil {
				m = &row{}
				mixed[pk] = m
				plans[k] = append(plans[k], pk.plan)
			}
			m.variant = r.Variant
			m.mixedGroups, m.mixedSl = cur.groups, cur.slots
			m.pairs, m.experts = cur.pairs, cur.experts
			m.parts = atoiOr(detailField(r.Detail, "parts"))
			m.crossPart = atoiOr(detailField(r.Detail, "crosspart"))
			m.widths = detailField(r.Detail, "widths")
			m.groupUs, _ = strconv.ParseFloat(detailField(r.Detail, "groupus"), 64)
			m.slotUs, _ = strconv.ParseFloat(detailField(r.Detail, "slotus"), 64)
			if mode == mode1 {
				m.mixedNs = r.NsPerIter
			} else {
				m.mixedBarNs = r.NsPerIter
			}
		case !v.mixed && mode == mode1 && v.weightsPerLoad == 32 && v.block == 128 && v.waveSize == 0:
			note(k)
			fixed[k] = append(fixed[k], cur)
		}
	}
	if len(keys) == 0 {
		return
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		if keys[i].tokens != keys[j].tokens {
			return keys[i].tokens < keys[j].tokens
		}
		return keys[i].nrows < keys[j].nrows
	})

	// Table 1: the rule against the oracle, at two calibrations.
	fmt.Fprintln(w, "\nsizing the M block from the routing histogram: the cost model's pick against the oracle")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tTOKENS\tNROWS\tROWS/EXPERT\tORACLE\tMS\tRULE@BATCH\tMS\tVS ORACLE\tRULE@t=4\tMS\tVS ORACLE\tGROUP US\tSLOT US\tGROUP/SLOT")
	for _, k := range keys {
		rows := fixed[k]
		if len(rows) < 2 {
			continue
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].mrows < rows[j].mrows })
		best := rows[0]
		for _, r := range rows {
			if r.ns < best.ns {
				best = r
			}
		}
		pick := func(c moeGEMVCost, ok bool) (row, bool) {
			if !ok {
				return row{}, false
			}
			out, first := row{}, true
			var bestCost float64
			for _, r := range rows {
				if p := c.predict(r.groups, r.slots); first || p < bestCost {
					out, bestCost, first = r, p, false
				}
			}
			return out, !first
		}
		atBatch, okB := fitMoEGEMVCost(results, k.layer, k.tokens, k.nrows, 32, 128, 0)
		atCal, okC := fitMoEGEMVCost(results, k.layer, moeSelectCalibrationTokens, k.nrows, 32, 128, 0)
		pb, hasB := pick(atBatch, okB)
		pc, hasC := pick(atCal, okC)
		cell := func(r row, has bool) (string, string, string) {
			if !has {
				return "-", "-", "-"
			}
			return fmt.Sprintf("MROWS=%d", r.mrows), fmt.Sprintf("%.3f", r.ns/1e6),
				fmt.Sprintf("%.2fx", best.ns/r.ns)
		}
		nb, mb, rb := cell(pb, hasB)
		nc, mc, rc := cell(pc, hasC)
		fmt.Fprintf(tw, "%s\t%d\t%d\t%.2f\tMROWS=%d\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.1f\n",
			shortLayer(k.layer), k.tokens, k.nrows, float64(best.pairs)/float64(max(best.experts, 1)),
			best.mrows, best.ns/1e6, nb, mb, rb, nc, mc, rc,
			atBatch.group/1e3, atBatch.slot/1e3, atBatch.ratio())
	}
	tw.Flush()
	fmt.Fprintln(w, "  the oracle is the fastest fixed width measured; the two rule columns are the width a")
	fmt.Fprintln(w, "  cost model picks off the counts alone, fitted at this batch and fitted once at t=4. VS")
	fmt.Fprintln(w, "  ORACLE below 1.00x is what choosing wrong costs. GROUP/SLOT is how many pad slots one")
	fmt.Fprintln(w, "  expert's weight read is worth at this batch, which is the number the decomposition turns on")

	// Table 2: the mixed-width dispatch.
	fmt.Fprintln(w, "\ncovering the histogram with several widths at once (one dispatch per width, same binaries)")
	tw = tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LAYER\tTOKENS\tNROWS\tBEST FIXED\tGROUPS\tSLOTS\tPAD\tMS\tPLAN\tPARTS\tGROUPS\tSLOTS\tPAD\tXPART\tMS\tVS FIXED\tPREDICTED\tMODEL\t+BARRIERS\tWIDTHS")
	for _, k := range keys {
		rows := fixed[k]
		if len(rows) == 0 {
			continue
		}
		best := rows[0]
		for _, r := range rows {
			if r.ns < best.ns {
				best = r
			}
		}
		names := plans[k]
		sort.Strings(names)
		for _, name := range names {
			m := mixed[planKey{k, name}]
			if m == nil || m.mixedNs <= 0 {
				continue
			}
			cost := moeGEMVCost{group: m.groupUs * 1e3, slot: m.slotUs * 1e3}
			predicted, model := "", ""
			if cost.ok() {
				p := cost.predict(m.mixedGroups, m.mixedSl)
				predicted = fmt.Sprintf("%.2fx", best.ns/p)
				model = fmt.Sprintf("%+.0f%%", 100*(p-m.mixedNs)/m.mixedNs)
			}
			bar := ""
			if m.mixedBarNs > 0 {
				bar = fmt.Sprintf("%+.1f%%", 100*(m.mixedBarNs-m.mixedNs)/m.mixedNs)
			}
			fmt.Fprintf(tw, "%s\t%d\t%d\tMROWS=%d\t%d\t%d\t%d\t%.3f\t%s\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.2fx\t%s\t%s\t%s\t%s\n",
				shortLayer(k.layer), k.tokens, k.nrows, best.mrows, best.groups, best.slots, best.pad, best.ns/1e6,
				name, m.parts, m.mixedGroups, m.mixedSl, m.mixedSl-m.pairs, m.crossPart, m.mixedNs/1e6,
				best.ns/m.mixedNs, predicted, model, bar, m.widths)
		}
	}
	tw.Flush()
	fmt.Fprintln(w, "  a plan is per-expert: cover c routed pairs with the widths that are built, paying one")
	fmt.Fprintln(w, "  group per cut and one slot per pad, and the widths' groups are bucketed so each is one")
	fmt.Fprintln(w, "  dispatch. `split` lets an expert take several widths (five pairs as a four and a one) and")
	fmt.Fprintln(w, "  `whole` gives it one width repeated, so XPART — an expert's groups that land in a part")
	fmt.Fprintln(w, "  other than its first — is zero for `whole` by construction and is where a `split` plan")
	fmt.Fprintln(w, "  re-reads an expert's weights a whole dispatch later instead of back to back.")
	fmt.Fprintln(w, "  PREDICTED is what the same cost model said the plan would be worth and MODEL is")
	fmt.Fprintln(w, "  how far off it was; +BARRIERS is what the parts cost when they are not allowed to overlap —")
	fmt.Fprintln(w, "  at PARTS=1 the two arms record the same commands, so that column is a reading of the noise")
}
