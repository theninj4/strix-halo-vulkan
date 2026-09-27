// Package ocr is PaddleOCR-VL-1.6 (OCR.md, O-stages): an image and a task
// prompt in, the recognised text out, from a NaViT vision tower, a 2x2-merge
// projector and ERNIE-4.5-0.3B.
//
// This file and its siblings are the host side (O1): the image processor
// (image.go), the chat template and its 3-D rope positions (prompt.go) and
// the tokenizer (tokenizer.go). Each is gated exactly against the HF
// processor and tokenizer through reference/dump_ocr.py and
// reference/dump_ocr_tokens.py.
package ocr
