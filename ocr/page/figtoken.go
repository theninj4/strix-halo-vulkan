package page

import (
	"math"
	"regexp"
	"strconv"
	"strings"

	"strix-halo-vulkan/llm/pixels"
)

// ---- paddleocr_vl/uilts.py tokenize_figure_of_table (OCR.md O10)
//
// A figure inside a table is painted over, in the table's crop, with a white
// box holding "[F<n>]" in OpenCV's Hershey simplex font, so the model writes
// the token where the picture sits; assembly swaps the token for an <img>.
// The painting is OpenCV 4.10's putText (drawing.cpp), ported for the
// calls paint_token makes: 8-bit, 3 channels, LINE_AA, a filled rectangle.
// <n> is a shuffle of numbers without the digits 0, 1 and 9 by Python's own
// random module, seeded with 1024 for every table.

// figToken is one painted figure: its token and its crop's markdown path.
type figToken struct {
	token, path string
}

// tokenizeFigures is tokenize_figure_of_table: the table crop with every
// figure inside it (at least 25 px a side) painted over, the tokens, and
// the paths of every figure inside it, painted or not (PaddleX drops those
// from the page).
func tokenizeFigures(img *pixels.RGB, table [4]int, figures []Box) (*pixels.RGB, []figToken, []string) {
	rmap := genRandomMap(len(figures))
	newPyRandom(1024).shuffle(rmap)
	var toks []figToken
	var dropped []string
	painted := false
	for i, f := range figures {
		c := f.Coord
		if c[0] < table[0] || c[1] < table[1] || c[2] > table[2] || c[3] > table[3] {
			continue
		}
		p := imgPath(f.Label, c)
		dropped = append(dropped, p)
		if min(c[2]-c[0], c[3]-c[1]) < 25 {
			continue
		}
		if !painted {
			img = &pixels.RGB{W: img.W, H: img.H, Pix: append([]uint8(nil), img.Pix...)}
			painted = true
		}
		tok := "[F" + strconv.Itoa(rmap[i]) + "]"
		paintToken(img, [4]int{c[0] - table[0], c[1] - table[1], c[2] - table[0], c[3] - table[1]}, tok)
		toks = append(toks, figToken{tok, p})
	}
	return img, toks, dropped
}

var figTokenRE = regexp.MustCompile(`\[F(\d+)\]`)

// untokenizeFigures is untokenize_figure_of_table: each token the model
// wrote back becomes the figure's <img> (and the figure block's content,
// if it has any), when the figure is one of the page's image blocks.
func untokenizeFigures(s string, toks []figToken, images map[string]string) string {
	return figTokenRE.ReplaceAllStringFunc(s, func(m string) string {
		path := m
		for _, t := range toks {
			if t.token == m {
				path = t.path
				break
			}
		}
		content, ok := images[path]
		if !ok {
			return m
		}
		info := `<img src="` + strings.ReplaceAll(strings.ReplaceAll(path, "-\n", ""), "\n", " ") + `" alt="Image"" />`
		if content != "" {
			info += "\n\n" + content + "\n\n"
		}
		return info
	})
}

// genRandomMap is gen_random_map: the first n numbers without a 0, 1 or 9.
func genRandomMap(n int) []int {
	var seq []int
	for i := 0; len(seq) < n; i++ {
		if !strings.ContainsAny(strconv.Itoa(i), "019") {
			seq = append(seq, i)
		}
	}
	return seq
}

// pyRandom is CPython's Mersenne Twister, for random.seed(int) and
// random.shuffle.
type pyRandom struct {
	mt  [624]uint32
	idx int
}

func newPyRandom(seed uint32) *pyRandom {
	r := &pyRandom{}
	// init_genrand(19650218) then init_by_array([seed]).
	r.mt[0] = 19650218
	for i := 1; i < 624; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	key := []uint32{seed}
	i, j := 1, 0
	for k := 624; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1] >> 30)) * 1664525)) + key[j] + uint32(j)
		i++
		j++
		if i >= 624 {
			r.mt[0] = r.mt[623]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k := 623; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1] >> 30)) * 1566083941)) - uint32(i)
		i++
		if i >= 624 {
			r.mt[0] = r.mt[623]
			i = 1
		}
	}
	r.mt[0] = 0x80000000
	r.idx = 624
	return r
}

func (r *pyRandom) uint32() uint32 {
	if r.idx >= 624 {
		for k := 0; k < 624; k++ {
			y := r.mt[k]&0x80000000 | r.mt[(k+1)%624]&0x7fffffff
			v := r.mt[(k+397)%624] ^ y>>1
			if y&1 != 0 {
				v ^= 0x9908b0df
			}
			r.mt[k] = v
		}
		r.idx = 0
	}
	y := r.mt[r.idx]
	r.idx++
	y ^= y >> 11
	y ^= y << 7 & 0x9d2c5680
	y ^= y << 15 & 0xefc60000
	y ^= y >> 18
	return y
}

// randBelow is _randbelow_with_getrandbits, for n < 2^32.
func (r *pyRandom) randBelow(n int) int {
	k := 0
	for v := n; v > 0; v >>= 1 {
		k++
	}
	for {
		if v := int(r.uint32() >> (32 - k)); v < n {
			return v
		}
	}
}

func (r *pyRandom) shuffle(x []int) {
	for i := len(x) - 1; i > 0; i-- {
		j := r.randBelow(i + 1)
		x[i], x[j] = x[j], x[i]
	}
}

// paintToken is paint_token: box filled white, token centred in black at
// the largest scale that fits 90% of the box's shorter side.
func paintToken(img *pixels.RGB, box [4]int, tok string) {
	x1, y1, x2, y2 := box[0], box[1], box[2], box[3]
	bw, bh := x2-x1, y2-y1
	fillRect(img, x1, y1, x2, y2, 255)

	sq := float64(min(bw, bh)) * 0.9
	left, right, scale := 0.2, 10.0, 0.2
	var tw, th int
	for right-left > 1e-2 {
		mid := (left + right) / 2
		tw, th = textSize(tok, mid, 1)
		if float64(tw) < sq && float64(th) < sq {
			scale, left = mid, mid
		} else {
			right = mid
		}
	}
	thick := max(1, int(math.Floor(scale*4)))
	tx := x1 + floorDiv(bw-tw, 2)
	ty := y1 + floorDiv(bh+th, 2)
	putText(img, tok, tx, ty, scale, thick)
}

func floorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// fillRect is cv2.rectangle(thickness=-1, LINE_8) with integer corners:
// both corners inclusive, clipped.
func fillRect(img *pixels.RGB, x1, y1, x2, y2 int, v uint8) {
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	x1, y1 = max(x1, 0), max(y1, 0)
	x2, y2 = min(x2, img.W-1), min(y2, img.H-1)
	for y := y1; y <= y2; y++ {
		for x := x1; x <= x2; x++ {
			o := (y*img.W + x) * 3
			img.Pix[o], img.Pix[o+1], img.Pix[o+2] = v, v, v
		}
	}
}

// Hershey simplex (HersheySimplex's header and the glyphs of the characters
// a token can hold, from hershey_fonts.cpp).
const hersheyBase, hersheyCap = 9, 12

var hersheyGlyphs = map[byte]string{
	'[': `KYOBOb OBVB ObVb`,
	']': `KYUBUb NBUB NbUb`,
	'F': `HZLFL[ LFYF LPTP`,
	'0': `H\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF`,
	'1': `H\NJPISFS[`,
	'2': `H\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[`,
	'3': `H\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW`,
	'4': `H\UFKTZT UFU[`,
	'5': `H\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW`,
	'6': `H\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT`,
	'7': `H\YFO[ KFYF`,
	'8': `H\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF`,
	'9': `H\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX`,
}

// cvRound is OpenCV's cvRound: to nearest, ties to even.
func cvRound(v float64) int64 { return int64(math.RoundToEven(v)) }

// textSize is cv::getTextSize for FONT_HERSHEY_SIMPLEX.
func textSize(text string, scale float64, thick int) (int, int) {
	h := cvRound(float64(hersheyCap+hersheyBase)*scale + float64((thick+1)/2))
	vx := 0.0
	for i := 0; i < len(text); i++ {
		g := hersheyGlyphs[text[i]]
		vx += float64(int(g[1])-int(g[0])) * scale
	}
	return int(cvRound(vx + float64(thick))), int(h)
}

const (
	xyShift = 16
	xyOne   = 1 << xyShift
)

type pt64 struct{ x, y int64 }

// putText is cv::putText for FONT_HERSHEY_SIMPLEX, black, LINE_AA, on an
// 8-bit 3-channel image, origin at the bottom left of the text.
func putText(img *pixels.RGB, text string, ox, oy int, scale float64, thick int) {
	hscale := cvRound(scale * xyOne)
	vscale := hscale
	viewX := int64(ox) << xyShift
	viewY := int64(oy)<<xyShift - hersheyBase*vscale
	for i := 0; i < len(text); i++ {
		g := hersheyGlyphs[text[i]]
		px, py := int64(g[0])-'R', int64(g[1])-'R'
		dx := py * hscale
		viewX -= px * hscale
		var pts []pt64
		for k := 2; ; {
			if k >= len(g) || g[k] == ' ' {
				if len(pts) > 1 {
					polyLine(img, pts, thick)
				}
				if k >= len(g) {
					break
				}
				k++
				pts = pts[:0]
				continue
			}
			pts = append(pts, pt64{(int64(g[k])-'R')*hscale + viewX, (int64(g[k+1])-'R')*vscale + viewY})
			k += 2
		}
		viewX += dx
	}
}

// polyLine is PolyLine, open, LINE_AA, shift XY_SHIFT.
func polyLine(img *pixels.RGB, v []pt64, thick int) {
	flags := 3
	p0 := v[0]
	for i := 1; i < len(v); i++ {
		thickLine(img, p0, v[i], thick, flags)
		p0 = v[i]
		flags = 2
	}
}

// thickLine is ThickLine at shift XY_SHIFT, LINE_AA.
func thickLine(img *pixels.RGB, p0, p1 pt64, thick, flags int) {
	if thick <= 1 {
		lineAA(img, p0, p1)
		return
	}
	dx := float64(p0.x-p1.x) / xyOne
	dy := float64(p1.y-p0.y) / xyOne
	r := dx*dx + dy*dy
	odd := thick & 1
	t := int64(thick) << (xyShift - 1)
	if math.Abs(r) > 2.220446049250313e-16 {
		r = (float64(t) + float64(odd)*xyOne*0.5) / math.Sqrt(r)
		d := pt64{cvRound(dy * r), cvRound(dx * r)}
		fillConvexPolyAA(img, []pt64{
			{p0.x + d.x, p0.y + d.y}, {p0.x - d.x, p0.y - d.y},
			{p1.x - d.x, p1.y - d.y}, {p1.x + d.x, p1.y + d.y},
		})
	}
	for i := 0; i < 2; i++ {
		if flags&(i+1) != 0 {
			filledCircleAA(img, p0, t)
		}
		p0 = p1
	}
}

// filledCircleAA is EllipseEx(center, (r, r), 0, 0, 360, thickness -1,
// LINE_AA) at shift XY_SHIFT.
func filledCircleAA(img *pixels.RGB, c pt64, r int64) {
	if r < 0 {
		r = -r
	}
	delta := (r + xyOne>>1) >> xyShift
	switch {
	case delta < 3:
		delta = 90
	case delta < 10:
		delta = 30
	case delta < 15:
		delta = 18
	default:
		delta = 5
	}
	var v []pt64
	prev := pt64{-1, -1}
	for i := int64(0); i < 360+delta; i += delta {
		a := min(i, 360)
		x := float64(c.x) + float64(r)*float64(sinTable[450-a])
		y := float64(c.y) + float64(r)*float64(sinTable[a])
		p := pt64{cvRound(x/xyOne) << xyShift, cvRound(y/xyOne) << xyShift}
		p.x += cvRound(x - float64(p.x))
		p.y += cvRound(y - float64(p.y))
		if p != prev {
			v = append(v, p)
			prev = p
		}
	}
	if len(v) <= 1 {
		v = []pt64{c, c}
	}
	fillConvexPolyAA(img, v)
}

// fillConvexPolyAA is FillConvexPoly at shift XY_SHIFT, LINE_AA: the edges
// by LineAA, the inside by whole spans.
func fillConvexPolyAA(img *pixels.RGB, v []pt64) {
	type edge struct {
		idx, di int
		x, dx   int64
		ye      int
	}
	const delta = xyOne >> 1
	const delta1, delta2 = xyOne - 1, 0
	npts := len(v)
	imin := 0
	xmin, xmax, ymin, ymax := v[0].x, v[0].x, v[0].y, v[0].y
	p0 := v[npts-1]
	for i, p := range v {
		if p.y < ymin {
			ymin, imin = p.y, i
		}
		ymax = max(ymax, p.y)
		xmax = max(xmax, p.x)
		xmin = min(xmin, p.x)
		lineAA(img, p0, p)
		p0 = p
	}
	xmin = (xmin + delta) >> xyShift
	xmax = (xmax + delta) >> xyShift
	ymin = (ymin + delta) >> xyShift
	ymax = (ymax + delta) >> xyShift
	if npts < 3 || int(xmax) < 0 || int(ymax) < 0 || int(xmin) >= img.W || int(ymin) >= img.H {
		return
	}
	ymax = min(ymax, int64(img.H-1))
	var e [2]edge
	e[0].idx, e[1].idx = imin, imin
	y := int(ymin)
	e[0].ye, e[1].ye = y, y
	e[0].di, e[1].di = 1, npts-1
	e[0].x, e[1].x = -xyOne, -xyOne
	edges := npts
	for {
		if y < int(ymax) || y == int(ymin) {
			for i := 0; i < 2; i++ {
				if y >= e[i].ye {
					idx0, di := e[i].idx, e[i].di
					idx := idx0 + di
					if idx >= npts {
						idx -= npts
					}
					for {
						// for (; edges-- > 0; )
						more := edges > 0
						edges--
						if !more {
							break
						}
						ty := int((v[idx].y + delta) >> xyShift)
						if ty > y {
							xs, xe := v[idx0].x, v[idx].x
							e[i].ye = ty
							e[i].dx = ((xe-xs)*2 + int64(ty-y)) / (2 * int64(ty-y))
							e[i].x = xs
							e[i].idx = idx
							break
						}
						idx0 = idx
						idx += di
						if idx >= npts {
							idx -= npts
						}
					}
				}
			}
		}
		if edges < 0 {
			break
		}
		if y >= 0 {
			l, r := 0, 1
			if e[0].x > e[1].x {
				l, r = 1, 0
			}
			xx1 := int((e[l].x + delta1) >> xyShift)
			xx2 := int((e[r].x + delta2) >> xyShift)
			if xx2 >= 0 && xx1 < img.W {
				xx1 = max(xx1, 0)
				xx2 = min(xx2, img.W-1)
				row := img.Pix[y*img.W*3:]
				for x := xx1; x <= xx2; x++ {
					row[x*3], row[x*3+1], row[x*3+2] = 0, 0, 0
				}
			}
		}
		e[0].x += e[0].dx
		e[1].x += e[1].dx
		y++
		if y > int(ymax) {
			break
		}
	}
}

var slopeCorrTable = [...]int64{
	181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201,
	203, 206, 209, 211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254,
}

var filterTable = [...]int64{
	168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254,
	254, 254, 252, 249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168,
	158, 149, 140, 131, 122, 114, 105, 97, 89, 82, 75, 68, 62, 56, 50, 45,
	40, 36, 32, 28, 25, 22, 19, 16, 14, 12, 11, 9, 8, 7, 5, 5,
}

// clipLine is clipLine(Size2l, Point2l&, Point2l&).
func clipLine(w, h int64, p1, p2 *pt64) bool {
	right, bottom := w-1, h-1
	if w <= 0 || h <= 0 {
		return false
	}
	code := func(p pt64) int {
		c := 0
		if p.x < 0 {
			c |= 1
		}
		if p.x > right {
			c |= 2
		}
		if p.y < 0 {
			c |= 4
		}
		if p.y > bottom {
			c |= 8
		}
		return c
	}
	xcode := func(p pt64) int {
		c := 0
		if p.x < 0 {
			c |= 1
		}
		if p.x > right {
			c |= 2
		}
		return c
	}
	c1, c2 := code(*p1), code(*p2)
	if c1&c2 == 0 && c1|c2 != 0 {
		if c1&12 != 0 {
			a := bottom
			if c1 < 8 {
				a = 0
			}
			p1.x += int64(float64(a-p1.y) * float64(p2.x-p1.x) / float64(p2.y-p1.y))
			p1.y = a
			c1 = xcode(*p1)
		}
		if c2&12 != 0 {
			a := bottom
			if c2 < 8 {
				a = 0
			}
			p2.x += int64(float64(a-p2.y) * float64(p2.x-p1.x) / float64(p2.y-p1.y))
			p2.y = a
			c2 = xcode(*p2)
		}
		if c1&c2 == 0 && c1|c2 != 0 {
			if c1 != 0 {
				a := right
				if c1 == 1 {
					a = 0
				}
				p1.y += int64(float64(a-p1.x) * float64(p2.y-p1.y) / float64(p2.x-p1.x))
				p1.x = a
				c1 = 0
			}
			if c2 != 0 {
				a := right
				if c2 == 1 {
					a = 0
				}
				p2.y += int64(float64(a-p2.x) * float64(p2.y-p1.y) / float64(p2.x-p1.x))
				p2.x = a
				c2 = 0
			}
		}
	}
	return c1|c2 == 0
}

// lineAA is LineAA for a 3-channel 8-bit image and colour (0, 0, 0).
func lineAA(img *pixels.RGB, pt1, pt2 pt64) {
	w0, h0 := img.W, img.H
	if !clipLine(int64(w0)<<xyShift, int64(h0)<<xyShift, &pt1, &pt2) {
		return
	}
	dx, dy := pt2.x-pt1.x, pt2.y-pt1.y
	var j, i int64
	if dx < 0 {
		j = -1
	}
	ax := (dx ^ j) - j
	if dy < 0 {
		i = -1
	}
	ay := (dy ^ i) - i
	var xStep, yStep int64
	var ecount, scount int
	var slope int64
	horizontal := ax > ay
	if horizontal {
		dy = (dy ^ j) - j
		pt1.x ^= pt2.x & j
		pt2.x ^= pt1.x & j
		pt1.x ^= pt2.x & j
		pt1.y ^= pt2.y & j
		pt2.y ^= pt1.y & j
		pt1.y ^= pt2.y & j
		xStep = xyOne
		yStep = int64(uint64(dy)<<xyShift) / (ax | 1)
		pt2.x += xyOne
		ecount = int((pt2.x >> xyShift) - (pt1.x >> xyShift))
		j = -(pt1.x & (xyOne - 1))
		pt1.y += ((yStep * j) >> xyShift) + (xyOne >> 1)
		slope = (yStep >> (xyShift - 5)) & 0x3f
		if yStep < 0 {
			slope ^= 0x3f
		}
		i = (pt1.x >> (xyShift - 7)) & 0x78
		j = (pt2.x >> (xyShift - 7)) & 0x78
	} else {
		dx = (dx ^ i) - i
		pt1.x ^= pt2.x & i
		pt2.x ^= pt1.x & i
		pt1.x ^= pt2.x & i
		pt1.y ^= pt2.y & i
		pt2.y ^= pt1.y & i
		pt1.y ^= pt2.y & i
		xStep = int64(uint64(dx)<<xyShift) / (ay | 1)
		yStep = xyOne
		pt2.y += xyOne
		ecount = int((pt2.y >> xyShift) - (pt1.y >> xyShift))
		j = -(pt1.y & (xyOne - 1))
		pt1.x += ((xStep * j) >> xyShift) + (xyOne >> 1)
		slope = (xStep >> (xyShift - 5)) & 0x3f
		if xStep < 0 {
			slope ^= 0x3f
		}
		i = (pt1.y >> (xyShift - 7)) & 0x78
		j = (pt2.y >> (xyShift - 7)) & 0x78
	}
	if slope&0x20 != 0 {
		slope = 0x100
	} else {
		slope = slopeCorrTable[slope]
	}
	var ep [9]int64
	{
		t0 := slope << 7
		t1 := ((0x78 - i) | 4) * slope
		t2 := (j | 4) * slope
		ep[0] = 0
		ep[8] = slope
		ep[1] = ((((j - i) & 0x78) | 4) * slope >> 8) & 0x1ff
		ep[3] = ep[1]
		ep[2] = (t1 >> 8) & 0x1ff
		ep[4] = ((((j - i) + 0x80) | 4) * slope >> 8) & 0x1ff
		ep[5] = ((t1 + t0) >> 8) & 0x1ff
		ep[6] = (t2 >> 8) & 0x1ff
		ep[7] = ((t2 + t0) >> 8) & 0x1ff
	}
	put := func(x, y int, a int64) {
		o := (y*w0 + x) * 3
		for c := 0; c < 3; c++ {
			v := int64(img.Pix[o+c])
			v += ((0-v)*a + 127) >> 8
			v += ((0-v)*a + 127) >> 8
			img.Pix[o+c] = uint8(v)
		}
	}
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return 0
	}
	epIdx := func() int {
		return ((b2i(scount >= 2)+1)&(scount|2))*3 + ((b2i(ecount >= 2) + 1) & (ecount | 2))
	}
	if horizontal {
		x := int(pt1.x >> xyShift)
		for ; ecount >= 0; x, pt1.y, scount, ecount = x+1, pt1.y+yStep, scount+1, ecount-1 {
			if uint(x) >= uint(w0) {
				continue
			}
			y := int((pt1.y >> xyShift) - 1)
			corr := ep[epIdx()]
			dist := (pt1.y >> (xyShift - 5)) & 31
			if uint(y) < uint(h0) {
				put(x, y, (corr*filterTable[dist+32]>>8)&0xff)
			}
			if uint(y+1) < uint(h0) {
				put(x, y+1, (corr*filterTable[dist]>>8)&0xff)
			}
			if uint(y+2) < uint(h0) {
				put(x, y+2, (corr*filterTable[63-dist]>>8)&0xff)
			}
		}
		return
	}
	y := int(pt1.y >> xyShift)
	for ; ecount >= 0; y, pt1.x, scount, ecount = y+1, pt1.x+xStep, scount+1, ecount-1 {
		if uint(y) >= uint(h0) {
			continue
		}
		x := int((pt1.x >> xyShift) - 1)
		corr := ep[epIdx()]
		dist := (pt1.x >> (xyShift - 5)) & 31
		if uint(x) < uint(w0) {
			put(x, y, (corr*filterTable[dist+32]>>8)&0xff)
		}
		if uint(x+1) < uint(w0) {
			put(x+1, y, (corr*filterTable[dist]>>8)&0xff)
		}
		if uint(x+2) < uint(w0) {
			put(x+2, y, (corr*filterTable[63-dist]>>8)&0xff)
		}
	}
}

// sinTable is drawing.cpp SinTable: sin of whole degrees, 0 to 450, as float.
var sinTable = [451]float32{
	0.0000000, 0.0174524, 0.0348995, 0.0523360, 0.0697565, 0.0871557,
	0.1045285, 0.1218693, 0.1391731, 0.1564345, 0.1736482, 0.1908090,
	0.2079117, 0.2249511, 0.2419219, 0.2588190, 0.2756374, 0.2923717,
	0.3090170, 0.3255682, 0.3420201, 0.3583679, 0.3746066, 0.3907311,
	0.4067366, 0.4226183, 0.4383711, 0.4539905, 0.4694716, 0.4848096,
	0.5000000, 0.5150381, 0.5299193, 0.5446390, 0.5591929, 0.5735764,
	0.5877853, 0.6018150, 0.6156615, 0.6293204, 0.6427876, 0.6560590,
	0.6691306, 0.6819984, 0.6946584, 0.7071068, 0.7193398, 0.7313537,
	0.7431448, 0.7547096, 0.7660444, 0.7771460, 0.7880108, 0.7986355,
	0.8090170, 0.8191520, 0.8290376, 0.8386706, 0.8480481, 0.8571673,
	0.8660254, 0.8746197, 0.8829476, 0.8910065, 0.8987940, 0.9063078,
	0.9135455, 0.9205049, 0.9271839, 0.9335804, 0.9396926, 0.9455186,
	0.9510565, 0.9563048, 0.9612617, 0.9659258, 0.9702957, 0.9743701,
	0.9781476, 0.9816272, 0.9848078, 0.9876883, 0.9902681, 0.9925462,
	0.9945219, 0.9961947, 0.9975641, 0.9986295, 0.9993908, 0.9998477,
	1.0000000, 0.9998477, 0.9993908, 0.9986295, 0.9975641, 0.9961947,
	0.9945219, 0.9925462, 0.9902681, 0.9876883, 0.9848078, 0.9816272,
	0.9781476, 0.9743701, 0.9702957, 0.9659258, 0.9612617, 0.9563048,
	0.9510565, 0.9455186, 0.9396926, 0.9335804, 0.9271839, 0.9205049,
	0.9135455, 0.9063078, 0.8987940, 0.8910065, 0.8829476, 0.8746197,
	0.8660254, 0.8571673, 0.8480481, 0.8386706, 0.8290376, 0.8191520,
	0.8090170, 0.7986355, 0.7880108, 0.7771460, 0.7660444, 0.7547096,
	0.7431448, 0.7313537, 0.7193398, 0.7071068, 0.6946584, 0.6819984,
	0.6691306, 0.6560590, 0.6427876, 0.6293204, 0.6156615, 0.6018150,
	0.5877853, 0.5735764, 0.5591929, 0.5446390, 0.5299193, 0.5150381,
	0.5000000, 0.4848096, 0.4694716, 0.4539905, 0.4383711, 0.4226183,
	0.4067366, 0.3907311, 0.3746066, 0.3583679, 0.3420201, 0.3255682,
	0.3090170, 0.2923717, 0.2756374, 0.2588190, 0.2419219, 0.2249511,
	0.2079117, 0.1908090, 0.1736482, 0.1564345, 0.1391731, 0.1218693,
	0.1045285, 0.0871557, 0.0697565, 0.0523360, 0.0348995, 0.0174524,
	0.0000000, -0.0174524, -0.0348995, -0.0523360, -0.0697565, -0.0871557,
	-0.1045285, -0.1218693, -0.1391731, -0.1564345, -0.1736482, -0.1908090,
	-0.2079117, -0.2249511, -0.2419219, -0.2588190, -0.2756374, -0.2923717,
	-0.3090170, -0.3255682, -0.3420201, -0.3583679, -0.3746066, -0.3907311,
	-0.4067366, -0.4226183, -0.4383711, -0.4539905, -0.4694716, -0.4848096,
	-0.5000000, -0.5150381, -0.5299193, -0.5446390, -0.5591929, -0.5735764,
	-0.5877853, -0.6018150, -0.6156615, -0.6293204, -0.6427876, -0.6560590,
	-0.6691306, -0.6819984, -0.6946584, -0.7071068, -0.7193398, -0.7313537,
	-0.7431448, -0.7547096, -0.7660444, -0.7771460, -0.7880108, -0.7986355,
	-0.8090170, -0.8191520, -0.8290376, -0.8386706, -0.8480481, -0.8571673,
	-0.8660254, -0.8746197, -0.8829476, -0.8910065, -0.8987940, -0.9063078,
	-0.9135455, -0.9205049, -0.9271839, -0.9335804, -0.9396926, -0.9455186,
	-0.9510565, -0.9563048, -0.9612617, -0.9659258, -0.9702957, -0.9743701,
	-0.9781476, -0.9816272, -0.9848078, -0.9876883, -0.9902681, -0.9925462,
	-0.9945219, -0.9961947, -0.9975641, -0.9986295, -0.9993908, -0.9998477,
	-1.0000000, -0.9998477, -0.9993908, -0.9986295, -0.9975641, -0.9961947,
	-0.9945219, -0.9925462, -0.9902681, -0.9876883, -0.9848078, -0.9816272,
	-0.9781476, -0.9743701, -0.9702957, -0.9659258, -0.9612617, -0.9563048,
	-0.9510565, -0.9455186, -0.9396926, -0.9335804, -0.9271839, -0.9205049,
	-0.9135455, -0.9063078, -0.8987940, -0.8910065, -0.8829476, -0.8746197,
	-0.8660254, -0.8571673, -0.8480481, -0.8386706, -0.8290376, -0.8191520,
	-0.8090170, -0.7986355, -0.7880108, -0.7771460, -0.7660444, -0.7547096,
	-0.7431448, -0.7313537, -0.7193398, -0.7071068, -0.6946584, -0.6819984,
	-0.6691306, -0.6560590, -0.6427876, -0.6293204, -0.6156615, -0.6018150,
	-0.5877853, -0.5735764, -0.5591929, -0.5446390, -0.5299193, -0.5150381,
	-0.5000000, -0.4848096, -0.4694716, -0.4539905, -0.4383711, -0.4226183,
	-0.4067366, -0.3907311, -0.3746066, -0.3583679, -0.3420201, -0.3255682,
	-0.3090170, -0.2923717, -0.2756374, -0.2588190, -0.2419219, -0.2249511,
	-0.2079117, -0.1908090, -0.1736482, -0.1564345, -0.1391731, -0.1218693,
	-0.1045285, -0.0871557, -0.0697565, -0.0523360, -0.0348995, -0.0174524,
	-0.0000000, 0.0174524, 0.0348995, 0.0523360, 0.0697565, 0.0871557,
	0.1045285, 0.1218693, 0.1391731, 0.1564345, 0.1736482, 0.1908090,
	0.2079117, 0.2249511, 0.2419219, 0.2588190, 0.2756374, 0.2923717,
	0.3090170, 0.3255682, 0.3420201, 0.3583679, 0.3746066, 0.3907311,
	0.4067366, 0.4226183, 0.4383711, 0.4539905, 0.4694716, 0.4848096,
	0.5000000, 0.5150381, 0.5299193, 0.5446390, 0.5591929, 0.5735764,
	0.5877853, 0.6018150, 0.6156615, 0.6293204, 0.6427876, 0.6560590,
	0.6691306, 0.6819984, 0.6946584, 0.7071068, 0.7193398, 0.7313537,
	0.7431448, 0.7547096, 0.7660444, 0.7771460, 0.7880108, 0.7986355,
	0.8090170, 0.8191520, 0.8290376, 0.8386706, 0.8480481, 0.8571673,
	0.8660254, 0.8746197, 0.8829476, 0.8910065, 0.8987940, 0.9063078,
	0.9135455, 0.9205049, 0.9271839, 0.9335804, 0.9396926, 0.9455186,
	0.9510565, 0.9563048, 0.9612617, 0.9659258, 0.9702957, 0.9743701,
	0.9781476, 0.9816272, 0.9848078, 0.9876883, 0.9902681, 0.9925462,
	0.9945219, 0.9961947, 0.9975641, 0.9986295, 0.9993908, 0.9998477,
	1.0000000,
}
