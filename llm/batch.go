package llm

// Batched decode (CONCURRENCY.md C5): several sequences advance one token
// each in one pass.
//
// Everything in a layer but three blocks is a function of the row alone: the
// hyper-connections, the MoE, the head and every projection read the same
// weights whatever sequence a row belongs to, so R rows cost what P5b's R-row
// GEMVs cost. The three that carry a sequence are the attention cache, the
// DeltaNet recurrence and the PLE ring. Each of them runs its per-sequence
// kernels once per row, as a one-token pass on that row's slot at that row's
// position, between projections that run over all R rows at once.
//
// The per-row position is the one thing no push field had room for, so it
// is a dword of the block's own arena: row r's is dword 1+r, and the row's
// dispatches name it in `lowRank` (SEQ_PAST is actu[pc.lowRank] in
// llm_common.glsl). An ordinary pass leaves `lowRank` at zero, which is
// dword 0, the position SetPast writes, so nothing about a single-sequence
// pass or its recorded replay changed.

import "fmt"

// batchRow is one row of a batched pass: the sequence slot it advances and
// that sequence's position before it.
type batchRow struct{ slot, past int }

// maxBatchRows bounds the per-row position table. Each block's arena starts
// with a 64-dword allocation for the position, of which only dword 0 was
// used, so the table fits without moving anything.
const maxBatchRows = 32

// checkBatch is what every block refuses: too many rows, a slot out of range
// or named twice (two rows of one sequence in a pass would each read the
// state the other is writing), or a position past the cache.
func checkBatch(rows []batchRow, slots, nKV int) error {
	if len(rows) > maxBatchRows {
		return fmt.Errorf("llm: a %d-row batch, the position table holds %d", len(rows), maxBatchRows)
	}
	seen := make(map[int]bool, len(rows))
	for _, r := range rows {
		if r.slot < 0 || r.slot >= slots {
			return fmt.Errorf("llm: batch row on slot %d of %d", r.slot, slots)
		}
		if seen[r.slot] {
			return fmt.Errorf("llm: slot %d twice in one batch", r.slot)
		}
		seen[r.slot] = true
		if r.past < 0 || r.past+1 > nKV {
			return fmt.Errorf("llm: a token at position %d of a %d-cell context", r.past, nKV)
		}
	}
	return nil
}

// positions is the position table's contents: row r's position at r.
func positions(rows []batchRow) []uint32 {
	w := make([]uint32, len(rows))
	for i, r := range rows {
		w[i] = uint32(r.past)
	}
	return w
}

// SetBatch makes the next passes batched over these rows, or ordinary again
// with nil. The pass's row count is still Resize's, and must be len(rows).
func (g *AttnGPU) SetBatch(rows []batchRow) error {
	if rows == nil {
		g.batch = nil
		return nil
	}
	if err := checkBatch(rows, g.slots, g.nKV); err != nil {
		return err
	}
	// A row's pass writes the context's pad rows past itself, up to the
	// widest query tile (see graph).
	if len(rows)+64 > roundUpInt(g.tokens, 64) {
		return fmt.Errorf("llm: a %d-row batch in arenas built for %d tokens", len(rows), g.tokens)
	}
	g.abuf.WriteUint32At(1, positions(rows))
	g.batch = rows
	return nil
}

func (g *DeltaNetGPU) SetBatch(rows []batchRow) error {
	if rows == nil {
		g.batch = nil
		return nil
	}
	if !g.seqSlots && len(rows) > 1 {
		return fmt.Errorf("llm: this block was staged with one sequence slot (GraphOpts.Slots)")
	}
	if err := checkBatch(rows, g.slots, int(^uint32(0)>>1)); err != nil {
		return err
	}
	g.abuf.WriteUint32At(1, positions(rows))
	g.batch = rows
	return nil
}

func (g *PLEGPU) SetBatch(rows []batchRow) error {
	if rows == nil {
		g.batch = nil
		return nil
	}
	if !g.seqSlots && len(rows) > 1 {
		return fmt.Errorf("llm: this block was staged with one sequence slot (GraphOpts.Slots)")
	}
	if err := checkBatch(rows, g.slots, int(^uint32(0)>>1)); err != nil {
		return err
	}
	g.abuf.WriteUint32At(1, positions(rows))
	g.batch = rows
	return nil
}
