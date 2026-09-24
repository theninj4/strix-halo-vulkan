package llm

import "fmt"

// From a rendered, tokenized prompt to the graph's Input (LLM-VISION.md V7).
//
// The template writes one `<|image_pad|>` per image (ImageMarkup), and HF's
// processor then replaces it with as many pads as the image has merged
// tokens before the text is tokenized. Here the order is reversed: the
// prompt is tokenized with its single pads, and each is widened in the ids.
// The pad is a special token, so it is one id wherever it appears and the two
// orders give the same stream (TestExpandImagesIsTheProcessor).

// PromptImage is one image of a prompt, in the order the conversation holds
// them: its merged grid, the tower's rows for it, and a hash of its pixels.
type PromptImage struct {
	GridH, GridW int
	Hash         uint64
	Embd         []float32
}

// ExpandImages widens the i-th pad in ids to images[i]'s GridH*GridW pads and
// returns the pass that runs them. The count has to agree: a pad the
// conversation's images do not account for is text that spelled the token,
// and the model would read a picture that is not there.
func ExpandImages(ids []int32, pad int32, images []PromptImage) (Input, error) {
	n := 0
	for _, id := range ids {
		if id == pad {
			n++
		}
	}
	if n != len(images) {
		return Input{}, fmt.Errorf("llm: the prompt holds %d image pads for %d images", n, len(images))
	}
	if n == 0 {
		return Input{IDs: ids}, nil
	}
	out := Input{IDs: make([]int32, 0, len(ids))}
	k := 0
	for _, id := range ids {
		if id != pad {
			out.IDs = append(out.IDs, id)
			continue
		}
		im := images[k]
		k++
		if im.GridH <= 0 || im.GridW <= 0 {
			return Input{}, fmt.Errorf("llm: image %d has a %dx%d grid", k-1, im.GridH, im.GridW)
		}
		out.Images = append(out.Images, InputImage{
			At: len(out.IDs), GridH: im.GridH, GridW: im.GridW, Hash: im.Hash, Embd: im.Embd,
		})
		for range im.GridH * im.GridW {
			out.IDs = append(out.IDs, pad)
		}
	}
	return out, nil
}
