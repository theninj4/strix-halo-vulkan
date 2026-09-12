package bench

import (
	"encoding/binary"
	"math"
)

// pcBuilder builds a push-constant byte blob matching a GLSL push_constant
// block of sequential 4-byte uint/float fields (no vectors/matrices, so no
// alignment padding to worry about).
type pcBuilder struct {
	buf []byte
}

func newPC() *pcBuilder { return &pcBuilder{} }

func (p *pcBuilder) U32(v uint32) *pcBuilder {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	p.buf = append(p.buf, b...)
	return p
}

func (p *pcBuilder) F32(v float32) *pcBuilder {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	p.buf = append(p.buf, b...)
	return p
}

func (p *pcBuilder) Bytes() []byte { return p.buf }
