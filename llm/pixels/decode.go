package pixels

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/gif" // registers the decoder; the first frame is what image.Decode returns
	_ "image/jpeg"
	_ "image/png"
)

// goJPEG converts JPEG with Go's own rules instead of libjpeg-turbo's. It
// exists for the measurement in decode_test.go and nothing else.
var goJPEG bool

// ErrUnsupported is a format this package recognizes and does not decode.
var ErrUnsupported = errors.New("pixels: unsupported image format")

// Decode reads an encoded image: PNG, JPEG or GIF (its first frame, which
// is what PIL's `Image.open` shows the processor).
//
// **WebP is refused**, not guessed at. The standard library has no decoder,
// and this module takes no dependencies. See LLM-VISION.md Q6.
//
// EXIF orientation is not applied, because the processor does not apply it:
// `Image.open` returns the stored pixels and nothing in transformers calls
// `exif_transpose`. A rotated phone photo is seen rotated by both.
//
// JPEG decodes to what Go's decoder produces, which is not always what
// libjpeg-turbo (Pillow's) produces. LLM-VISION.md V3 has the measurement.
func Decode(data []byte) (*RGB, string, error) {
	if len(data) >= 12 && string(data[0:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		return nil, "webp", fmt.Errorf("%w: webp is not decoded here; send PNG, JPEG or GIF", ErrUnsupported)
	}
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, "", fmt.Errorf("pixels: decoding the image: %w", err)
	}
	if p, ok := img.(*image.Paletted); ok && format == "gif" {
		restoreTransparent(p, data)
	}
	if y, ok := img.(*image.YCbCr); ok && !goJPEG {
		if rgb := fromYCbCr(y); rgb != nil {
			return rgb, format, nil
		}
	}
	return FromImage(img), format, nil
}

// restoreTransparent puts back the colour of a GIF's transparent palette
// entry. Go's decoder replaces it with (0, 0, 0, 0), while PIL keeps the
// stored RGB and convert("RGB") shows it. The table the first frame uses is
// read from the file: its local table if it has one, the global one
// otherwise. A file this walk cannot follow is left as Go decoded it.
func restoreTransparent(p *image.Paletted, data []byte) {
	if len(data) < 13 {
		return
	}
	var global []byte
	i := 13
	if data[10]&0x80 != 0 {
		n := 3 << (data[10]&7 + 1)
		if i+n > len(data) {
			return
		}
		global, i = data[i:i+n], i+n
	}
	for i < len(data) {
		switch data[i] {
		case 0x21: // an extension: label, then sub-blocks until a zero length
			i += 2
			for i < len(data) && data[i] != 0 {
				i += int(data[i]) + 1
			}
			i++
		case 0x2C: // the first image descriptor
			table := global
			if i+10 <= len(data) && data[i+9]&0x80 != 0 {
				n := 3 << (data[i+9]&7 + 1)
				if i+10+n > len(data) {
					return
				}
				table = data[i+10 : i+10+n]
			}
			for k, c := range p.Palette {
				if _, _, _, a := c.RGBA(); a == 0 && 3*k+2 < len(table) {
					p.Palette[k] = color.NRGBA{table[3*k], table[3*k+1], table[3*k+2], 0}
				}
			}
			return
		default:
			return
		}
	}
}
