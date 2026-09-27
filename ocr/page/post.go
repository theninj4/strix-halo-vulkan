package page

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// pyIsSpace is Python's str.isspace for one code point, which strip() uses:
// Go's unicode.IsSpace misses the four ASCII separators \x1c-\x1f.
func pyIsSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

func pyStrip(s string) string { return strings.TrimFunc(s, pyIsSpace) }

// truncateRepetitive is paddleocr_vl/uilts.py truncate_repetitive_content:
// a long result that ends in a phrase repeated past half its length is cut
// before the repeats; a result that is one unit repeated ten times or more
// becomes the unit; one line making up 80% of ten or more lines becomes
// that line. Lengths are Python's, in code points.
func truncateRepetitive(content string, minCount int) string {
	const lineThreshold, charThreshold, minLen = 10, 10, 10
	if len([]rune(content)) < minCount {
		return content
	}
	stripped := pyStrip(content)
	if stripped == "" {
		return content
	}
	rs := []rune(stripped)
	if !strings.Contains(stripped, "\n") && len(rs) > 100 {
		if prefix, unit, count, ok := repeatingSuffix(rs, 8, 5); ok {
			if float64(len(unit)*count) > float64(len(rs))*0.5 {
				return string(prefix)
			}
		}
	}
	if !strings.Contains(stripped, "\n") && len(rs) > minLen {
		if unit := shortestRepeating(rs); unit != nil {
			if len(rs)/len(unit) >= charThreshold {
				return string(unit)
			}
		}
	}
	var lines []string
	for _, l := range strings.Split(content, "\n") {
		if t := pyStrip(l); t != "" {
			lines = append(lines, t)
		}
	}
	if len(lines) < lineThreshold {
		return content
	}
	counts := map[string]int{}
	var order []string
	for _, l := range lines {
		if counts[l] == 0 {
			order = append(order, l)
		}
		counts[l]++
	}
	best := order[0]
	for _, l := range order {
		if counts[l] > counts[best] {
			best = l
		}
	}
	if counts[best] >= lineThreshold && float64(counts[best])/float64(len(lines)) >= 0.8 {
		return best
	}
	return content
}

func runesEqual(a, b []rune) bool {
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

// shortestRepeating is find_shortest_repeating_substring.
func shortestRepeating(s []rune) []rune {
	n := len(s)
	for i := 1; i <= n/2; i++ {
		if n%i != 0 {
			continue
		}
		ok := true
		for k := i; k < n && ok; k += i {
			ok = runesEqual(s[k:k+i], s[:i])
		}
		if ok {
			return s[:i]
		}
	}
	return nil
}

// repeatingSuffix is find_repeating_suffix: the longest unit (at most a
// fifth of the string, at least minLen) the string ends with minRepeats
// times, and how many times it ends with it.
func repeatingSuffix(s []rune, minLen, minRepeats int) ([]rune, []rune, int, bool) {
	endsWith := func(t, u []rune) bool { return len(t) >= len(u) && runesEqual(t[len(t)-len(u):], u) }
	for i := len(s) / minRepeats; i >= minLen; i-- {
		unit := s[len(s)-i:]
		rep := make([]rune, 0, i*minRepeats)
		for k := 0; k < minRepeats; k++ {
			rep = append(rep, unit...)
		}
		if !endsWith(s, rep) {
			continue
		}
		count, t := 0, s
		for endsWith(t, unit) {
			t = t[:len(t)-i]
			count++
		}
		return s[:len(s)-count*i], unit, count, true
	}
	return nil, nil, 0, false
}

// formulaDelimiters is the pipeline's LaTeX delimiter rewrite: a result with
// \( \) or \[ \] loses its dollars, then inline and display delimiters
// become $ and $$ (a doubled \[\[ or \]\] once), and a formula number keeps
// no dollars at all.
func formulaDelimiters(r, label string) string {
	if !((strings.Contains(r, `\(`) && strings.Contains(r, `\)`)) || (strings.Contains(r, `\[`) && strings.Contains(r, `\]`))) {
		return r
	}
	r = strings.ReplaceAll(r, "$", "")
	for _, p := range [][2]string{{`\(`, " $ "}, {`\)`, " $"}, {`\[\[`, `\[`}, {`\]\]`, `\]`}, {`\[`, " $$ "}, {`\]`, " $$ "}} {
		r = strings.ReplaceAll(r, p[0], p[1])
	}
	if label == "formula_number" {
		r = strings.ReplaceAll(r, "$", "")
	}
	return r
}

// OTSL, paddleocr_vl/uilts.py: the table markup the model writes.
const (
	otslNL   = "<nl>"
	otslFCEL = "<fcel>"
	otslECEL = "<ecel>"
	otslLCEL = "<lcel>"
	otslUCEL = "<ucel>"
	otslXCEL = "<xcel>"
)

var otslTags = []string{otslNL, otslFCEL, otslECEL, otslLCEL, otslUCEL, otslXCEL}

func isTag(s string) bool {
	for _, t := range otslTags {
		if s == t {
			return true
		}
	}
	return false
}

// tagAt is the OTSL tag starting at s[i:], or "".
func tagAt(s string, i int) string {
	for _, t := range otslTags {
		if strings.HasPrefix(s[i:], t) {
			return t
		}
	}
	return ""
}

// otslCells is OTSL_FIND_PATTERN.findall on one line: from each tag to the
// next tag, or to the end (before a final newline, as Python's $ allows).
func otslCells(line string) []string {
	var starts []int
	for i := 0; i < len(line); {
		if t := tagAt(line, i); t != "" {
			starts = append(starts, i)
			i += len(t)
			continue
		}
		i++
	}
	var out []string
	for k, st := range starts {
		end := len(line)
		if k+1 < len(starts) {
			end = starts[k+1]
		} else if strings.HasSuffix(line, "\n") && end-1 >= st+len(tagAt(line, st)) {
			end--
		}
		out = append(out, line[st:end])
	}
	return out
}

// otslPadToSquare is otsl_pad_to_sqr_v2: every row cut or padded with
// empty cells to the width that costs least, at least as wide as the
// widest row's last filled cell.
func otslPadToSquare(s string) string {
	s = pyStrip(s)
	if !strings.Contains(s, otslNL) {
		return s + otslNL
	}
	type rowData struct {
		cells           []string
		total, minWidth int
	}
	var rows []rowData
	for _, line := range strings.Split(s, otslNL) {
		if line == "" {
			continue
		}
		cells := otslCells(line)
		if len(cells) == 0 {
			continue
		}
		minW := 0
		for i, c := range cells {
			if strings.HasPrefix(c, otslFCEL) {
				minW = i + 1
			}
		}
		rows = append(rows, rowData{cells, len(cells), minW})
	}
	if len(rows) == 0 {
		return otslNL
	}
	gmin, gmax := 0, 0
	for _, r := range rows {
		gmin, gmax = max(gmin, r.minWidth), max(gmax, r.total)
	}
	end := max(gmin, gmax)
	best, width := -1, end
	for w := gmin; w <= end; w++ {
		cost := 0
		for _, r := range rows {
			cost += absInt(r.total - w)
		}
		if best < 0 || cost < best {
			best, width = cost, w
		}
	}
	var lines []string
	for _, r := range rows {
		c := r.cells
		if len(c) > width {
			c = c[:width]
		} else {
			for len(c) < width {
				c = append(c, otslECEL)
			}
		}
		lines = append(lines, strings.Join(c, ""))
	}
	return strings.Join(lines, otslNL) + otslNL
}

type tableCell struct {
	rowSpan, colSpan int
	r0, r1, c0, c1   int
	text             string
}

// otslToHTML is convert_otsl_to_html.
func otslToHTML(s string) string {
	s = otslPadToSquare(s)
	// otsl_extract_tokens_and_text: the tags, and re.split's parts (tags
	// included) that are not blank.
	var tokens, texts []string
	last := 0
	for i := 0; i < len(s); {
		if t := tagAt(s, i); t != "" {
			if p := s[last:i]; pyStrip(p) != "" {
				texts = append(texts, p)
			}
			tokens = append(tokens, t)
			texts = append(texts, t)
			i += len(t)
			last = i
			continue
		}
		i++
	}
	if p := s[last:]; pyStrip(p) != "" {
		texts = append(texts, p)
	}

	// otsl_parse_texts: rows split at <nl>, squared with <ecel>.
	var rowsT [][]string
	var cur []string
	started := false
	for _, t := range tokens {
		if t == otslNL {
			if started {
				rowsT = append(rowsT, cur)
			}
			cur, started = nil, false
			continue
		}
		cur, started = append(cur, t), true
	}
	if started {
		rowsT = append(rowsT, cur)
	}
	if len(rowsT) > 0 {
		maxCols := 0
		for _, r := range rowsT {
			maxCols = max(maxCols, len(r))
		}
		for i := range rowsT {
			for len(rowsT[i]) < maxCols {
				rowsT[i] = append(rowsT[i], otslECEL)
			}
		}
		var nt []string
		ti := 0
		for _, row := range rowsT {
			for _, tok := range row {
				nt = append(nt, tok)
				if ti < len(texts) && texts[ti] == tok {
					ti++
					if ti < len(texts) && !isTag(texts[ti]) {
						nt = append(nt, texts[ti])
						ti++
					}
				}
			}
			nt = append(nt, otslNL)
			if ti < len(texts) && texts[ti] == otslNL {
				ti++
			}
		}
		texts = nt
	}
	// The two counters index past a row's end where Python would raise; a
	// malformed table stops counting there instead.
	countRight := func(c, r int, which map[string]bool) int {
		span := 0
		for c < len(rowsT[r]) && which[rowsT[r][c]] {
			c++
			span++
			if c >= len(rowsT[r]) {
				return span
			}
		}
		return span
	}
	countDown := func(c, r int, which map[string]bool) int {
		span := 0
		for r < len(rowsT) && c < len(rowsT[r]) && which[rowsT[r][c]] {
			r++
			span++
			if r >= len(rowsT) {
				return span
			}
		}
		return span
	}
	lx := map[string]bool{otslLCEL: true, otslXCEL: true}
	ux := map[string]bool{otslUCEL: true, otslXCEL: true}
	var cells []tableCell
	r, c := 0, 0
	for i, t := range texts {
		if t == otslFCEL || t == otslECEL {
			rs, cs, off := 1, 1, 1
			text := ""
			if t != otslECEL {
				text = texts[i+1]
				off = 2
			}
			right := ""
			if i+off < len(texts) {
				right = texts[i+off]
			}
			bottom := ""
			if r+1 < len(rowsT) && c < len(rowsT[r+1]) {
				bottom = rowsT[r+1][c]
			}
			if lx[right] {
				cs += countRight(c+1, r, lx)
			}
			if ux[bottom] {
				rs += countDown(c, r+1, ux)
			}
			cells = append(cells, tableCell{rowSpan: rs, colSpan: cs, r0: r, r1: r + rs, c0: c, c1: c + cs, text: pyStrip(text)})
		}
		if t == otslFCEL || t == otslECEL || t == otslLCEL || t == otslUCEL || t == otslXCEL {
			c++
		}
		if t == otslNL {
			r++
			c = 0
		}
	}

	// export_to_html over TableData's grid.
	nrows, ncols := len(rowsT), 0
	for _, row := range rowsT {
		ncols = max(ncols, len(row))
	}
	if len(cells) == 0 {
		return ""
	}
	grid := make([][]*tableCell, nrows)
	for i := range grid {
		grid[i] = make([]*tableCell, ncols)
		for j := range grid[i] {
			grid[i][j] = &tableCell{rowSpan: 1, colSpan: 1, r0: i, r1: i + 1, c0: j, c1: j + 1}
		}
	}
	for k := range cells {
		cl := &cells[k]
		for i := min(cl.r0, nrows); i < min(cl.r1, nrows); i++ {
			for j := min(cl.c0, ncols); j < min(cl.c1, ncols); j++ {
				grid[i][j] = cl
			}
		}
	}
	var b strings.Builder
	for i := 0; i < nrows; i++ {
		b.WriteString("<tr>")
		for j := 0; j < ncols; j++ {
			cl := grid[i][j]
			if cl.r0 != i || cl.c0 != j {
				continue
			}
			tag := "td"
			if cl.rowSpan > 1 {
				tag += fmt.Sprintf(` rowspan="%d"`, cl.rowSpan)
			}
			if cl.colSpan > 1 {
				tag += fmt.Sprintf(` colspan="%d"`, cl.colSpan)
			}
			fmt.Fprintf(&b, "<%s>%s</td>", tag, htmlEscape(pyStrip(cl.text)))
		}
		b.WriteString("</tr>")
	}
	return "<table>" + b.String() + "</table>"
}

// htmlEscape is Python's html.escape with quote=True.
func htmlEscape(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&#x27;").Replace(s)
}

// ---- markdown, common/result/converter/markdown_format_funcs.py (pretty)

func collapseSoftNewlines(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "-\n", ""), "\n", " ")
}

func centered(s string) string {
	return `<div style="text-align: center;">` + collapseSoftNewlines(s) + "</div>\n"
}

// titleRE is compile_title_pattern, with Python's $ (which also matches
// before a final newline) written out.
var titleRE = regexp.MustCompile(`^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|[(（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[)）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、\.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)\n?$`)

// formatTitle is format_title.
func formatTitle(content string) string {
	title := content
	if m := titleRE.FindStringSubmatch(title); m != nil {
		title = pyStrip(m[1]) + " " + strings.TrimLeftFunc(m[3], pyIsSpace)
	}
	title = strings.TrimRight(title, ".")
	level := 1
	if strings.Contains(title, ".") {
		level = strings.Count(title, ".") + 1
	}
	return collapseSoftNewlines("#" + strings.Repeat("#", level) + " " + title)
}

// formatFirstLine is format_first_line.
func formatFirstLine(content string, templates []string, f func(string) string, sep string) string {
	lines := strings.Split(content, sep)
	for i, l := range lines {
		if pyStrip(l) == "" {
			continue
		}
		lower := strings.ToLower(l)
		for _, t := range templates {
			if lower == t {
				lines[i] = f(l)
				break
			}
		}
		break
	}
	return strings.Join(lines, sep)
}

// Result is one block of the parsed page (PaddleOCRVLBlock).
type Result struct {
	Label   string
	BBox    [4]int
	Content string
	// GroupID is the merged group's first block, or -1.
	GroupID int
	// Image is the crop's markdown path (construct_img_path) for an image,
	// seal or chart region, else "".
	Image string
	Img   interface{} // the crop (*pixels.RGB) for an image region
}

// markdownIgnore is the pipeline config's markdown_ignore_labels.
var markdownIgnore = map[string]bool{"number": true, "footnote": true, "header": true, "header_image": true,
	"footer": true, "footer_image": true, "aside_text": true}

// Markdown is MarkdownConverter.convert with PaddleOCR-VL's handlers,
// pretty, formula numbers not merged, for a page w pixels wide.
func Markdown(blocks []Result, width int) string { return markdown(blocks, width, true, nil) }

// MarkdownPlain is the same with pretty=False: images as ![](path), text
// and titles without the centring HTML, tables unstyled.
func MarkdownPlain(blocks []Result) string { return markdown(blocks, 0, false, nil) }

// MistralTable is a table taken out of the markdown (Mistral's
// table_format "html").
type MistralTable struct {
	ID, HTML string
}

// MistralImage is a picture the markdown references, by Mistral's id.
type MistralImage struct {
	ID  string
	Box [4]int
	Img interface{} // *pixels.RGB
}

// mistralRefs renumbers a page's references Mistral's way.
type mistralRefs struct {
	separateTables bool
	img, tbl       int
	images         []MistralImage
	tables         []MistralTable
}

// Mistral is the plain markdown with Mistral's references: each picture
// ![img-N.jpeg](img-N.jpeg) and, with separateTables, each table
// [tbl-N.html](tbl-N.html) with its HTML returned apart. N counts from
// imgStart and tblStart, so that a document's pages number on.
func Mistral(blocks []Result, separateTables bool, imgStart, tblStart int) (string, []MistralTable, []MistralImage) {
	r := &mistralRefs{separateTables: separateTables, img: imgStart, tbl: tblStart}
	md := markdown(blocks, 0, false, r)
	return md, r.tables, r.images
}

func markdown(blocks []Result, width int, pretty bool, refs *mistralRefs) string {
	image := func(b Result) string {
		if b.Image == "" {
			return ""
		}
		scale := int(float64(b.BBox[2]-b.BBox[0]) / float64(width) * 100)
		return centered(fmt.Sprintf(`<img src="%s" alt="Image" width="%d%%" />`, collapseSoftNewlines(b.Image), scale))
	}
	para := func(b Result) string {
		return strings.ReplaceAll(strings.ReplaceAll(b.Content, "\n\n", "\n"), "\n", "\n\n")
	}
	text := func(b Result) string { return centered(b.Content) }
	plain := func(b Result) string { return b.Content }
	if !pretty {
		image = func(b Result) string {
			if b.Image == "" {
				return ""
			}
			return "![](" + collapseSoftNewlines(b.Image) + ")"
		}
		text = plain
	}
	table := func(b Result) string {
		return "\n" + strings.NewReplacer(
			"<table>", "<table border=1 style='margin: auto; word-wrap: break-word;'>",
			"<th>", "<th style='text-align: center; word-wrap: break-word;'>",
			"<td>", "<td style='text-align: center; word-wrap: break-word;'>").Replace(b.Content)
	}
	if !pretty {
		// simplify_table("\n" + content).
		table = func(b Result) string {
			return "\n\n" + strings.NewReplacer("<html>", "", "</html>", "", "<body>", "", "</body>", "").Replace(b.Content)
		}
	}
	if refs != nil {
		image = func(b Result) string {
			if b.Image == "" {
				return ""
			}
			id := fmt.Sprintf("img-%d.jpeg", refs.img)
			refs.img++
			refs.images = append(refs.images, MistralImage{ID: id, Box: b.BBox, Img: b.Img})
			return "![" + id + "](" + id + ")"
		}
		if refs.separateTables {
			table = func(b Result) string {
				id := fmt.Sprintf("tbl-%d.html", refs.tbl)
				refs.tbl++
				refs.tables = append(refs.tables, MistralTable{ID: id, HTML: b.Content})
				return "[" + id + "](" + id + ")"
			}
		}
	}
	handlers := map[string]func(Result) string{
		"paragraph_title": func(b Result) string { return formatTitle(b.Content) },
		"abstract_title":  func(b Result) string { return formatTitle(b.Content) },
		"reference_title": func(b Result) string { return formatTitle(b.Content) },
		"content_title":   func(b Result) string { return formatTitle(b.Content) },
		"doc_title":       func(b Result) string { return collapseSoftNewlines("# " + b.Content) },
		"table_title":     text, "figure_title": text, "chart_title": text,
		"vision_footnote": para, "text": para, "ocr": para, "vertical_text": para, "reference_content": para,
		"abstract": func(b Result) string {
			return formatFirstLine(b.Content, []string{"摘要", "abstract"}, func(l string) string { return "## " + l + "\n" }, " ")
		},
		"content": func(b Result) string {
			return strings.ReplaceAll(strings.ReplaceAll(b.Content, "-\n", "  \n"), "\n", "  \n")
		},
		"image": image, "chart": image, "seal": image, "header_image": image, "footer_image": image,
		"formula": plain, "display_formula": plain, "inline_formula": plain,
		"table": table,
		"reference": func(b Result) string {
			return formatFirstLine(b.Content, []string{"参考文献", "references"}, func(l string) string { return "## " + l }, "\n")
		},
		"algorithm": func(b Result) string { return strings.Trim(b.Content, "\n") },
		"spotting":  plain, "number": plain, "footnote": plain, "header": plain, "footer": plain, "aside_text": plain,
	}
	var md string
	for _, b := range blocks {
		f, ok := handlers[b.Label]
		if !ok || markdownIgnore[b.Label] {
			continue
		}
		if md != "" {
			md += "\n\n" + f(b)
		} else {
			md = f(b)
		}
	}
	return md
}
