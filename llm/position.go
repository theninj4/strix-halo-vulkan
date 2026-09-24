package llm

import "fmt"

// Rotary positions once a sequence can hold an image (LLM-VISION.md V5).
//
// On text a token's rotary position is its cell, and every piece of this
// package was written on that identity: the device's rotary table is indexed
// by cell, the pack ropes cell SEQ_PAST + t, and a pooled indexer block is
// roped at its first cell. An image breaks the identity, in the way HF's
// `get_rope_index` defines. An image of gh x gw merged tokens starting at text
// position p gives its token (r, c) the position (t, h, w) = (p, p + r, p + c),
// in raster order, and the text after it resumes at p + max(gh, gw), not at
// p + gh*gw. So past an image, cell and position part by a per-sequence
// delta.
//
// The fix is where HF's is: HF indexes its cos/sin rows by cell (the indexer
// ropes a pooled block with `full_cos.index_select(0, group_starts)`, and
// group_starts are cells), and so does this package. Only what a row holds
// changes. Row c is the angles of cell c's (t, h, w). No shader reads a
// position; each reads a table row by cell. The table becomes per slot and is
// rewritten, a pass at a time, only when a sequence needs non-default rows.

// Pos3 is a token's rotary position: temporal, height and width. On text
// all three are the same number.
type Pos3 [3]int32

// TextPos is the position of a text token at position p.
func TextPos(p int) Pos3 { return Pos3{int32(p), int32(p), int32(p)} }

// ImageSpan is one image's cells in a sequence: [Start, End) holds its
// GridH x GridW merged tokens in raster order. Hash identifies the pixels,
// because the ids an image occupies are all the same pad token, and two
// different images must never pass for one another (a restore, a prefix).
type ImageSpan struct {
	Start, End   int
	GridH, GridW int
	Hash         uint64
}

// Advance is how many positions the image consumes: max(gh, gw), not its
// token count.
func (s ImageSpan) Advance() int { return max(s.GridH, s.GridW) }

func (s ImageSpan) valid() error {
	if s.GridH <= 0 || s.GridW <= 0 || s.End-s.Start != s.GridH*s.GridW || s.Start < 0 {
		return fmt.Errorf("llm: an image span [%d, %d) for a %dx%d grid", s.Start, s.End, s.GridH, s.GridW)
	}
	return nil
}

// cellDelta is position minus cell for a text cell at `cell`, given the
// sequence's spans in order: each image before it shortens the positions by
// its token count less its advance.
func cellDelta(spans []ImageSpan, cell int) int {
	d := 0
	for _, s := range spans {
		if s.End > cell {
			break
		}
		d += s.Advance() - (s.End - s.Start)
	}
	return d
}

// cellPositions is the (t, h, w) of cells [from, from+n).
func cellPositions(spans []ImageSpan, from, n int) []Pos3 {
	out := make([]Pos3, n)
	for i := range out {
		out[i] = cellPosition(spans, from+i)
	}
	return out
}

func cellPosition(spans []ImageSpan, cell int) Pos3 {
	d := 0
	for _, s := range spans {
		if cell < s.Start {
			break
		}
		if cell < s.End {
			p0 := int32(s.Start + d) // the image's text position
			k := cell - s.Start
			return Pos3{p0, p0 + int32(k/s.GridW), p0 + int32(k%s.GridW)}
		}
		d += s.Advance() - (s.End - s.Start)
	}
	return TextPos(cell + d)
}

// truncateSpans is the spans a sequence cut back to `past` cells keeps. Cutting
// inside an image is refused: its tokens were computed as one picture, and
// half of one is not a sequence the model was shown.
func truncateSpans(spans []ImageSpan, past int) ([]ImageSpan, error) {
	for i, s := range spans {
		if s.Start >= past {
			return spans[:i], nil
		}
		if s.End > past {
			return nil, fmt.Errorf("llm: position %d falls inside an image at cells [%d, %d)", past, s.Start, s.End)
		}
	}
	return spans, nil
}
