package g2p

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>

// espeak-ng's three entry points, declared here rather than included: the
// library ships without headers in the place this project finds it, and three
// signatures are cheaper to write down than a build-time dependency.
typedef int (*espeak_init_fn)(int, int, const char *, int);
typedef int (*espeak_voice_fn)(const char *);
typedef const char *(*espeak_ttp_fn)(const void **, int, int);

static int espeak_call_init(void *f, int out, int buflen, const char *path, int opts) {
	return ((espeak_init_fn)f)(out, buflen, path, opts);
}
static int espeak_call_voice(void *f, const char *name) {
	return ((espeak_voice_fn)f)(name);
}
static const char *espeak_call_ttp(void *f, void **textptr, int tm, int pm) {
	return ((espeak_ttp_fn)f)((const void **)textptr, tm, pm);
}
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"
)

// Espeak is the fallback for a word the dictionary cannot answer for
// (SPEECH.md T5d), which over real running English is 8.8% of tokens — proper
// nouns, borrowings, invented words and acronyms espeak reads better than a
// spelling-out does.
//
// The library is opened with `dlopen` at run time rather than linked, for two
// reasons. `go build` then works on a machine that has no espeak at all, which
// matters because everything else in this repository does; and the copy this
// project actually uses is the one bundled inside misaki's wheel, whose path
// is a Python version away from being wrong. `Open` searches, and a caller
// that gets an error simply has no fallback — which is what the engine did
// before T5d and still degrades to.
//
// espeak-ng keeps global state and is not reentrant, so one process has one
// voice at a time and every call takes the lock.
type Espeak struct {
	mu      sync.Mutex
	handle  unsafe.Pointer
	init    unsafe.Pointer
	voice   unsafe.Pointer
	ttp     unsafe.Pointer
	british bool
	Library string
	Data    string
}

// The phoneme mode phonemizer asks for: IPA, with a tie character between the
// halves of a single phoneme, and that character passed in the high bits.
//
// The tie is what makes misaki's rewrite table expressible at all — without it
// the `eɪ` of one diphthong and the `eɪ` that spans a syllable boundary are
// the same two characters.
const (
	espeakIPA     = 0x02
	espeakTieFlag = 0x01 << 7
	espeakTie     = '͡'
	espeakUTF8    = 1
	espeakSynch   = 0x02 // AUDIO_OUTPUT_SYNCHRONOUS: nothing is played
)

// punctMarks is phonemizer's default set. A word containing one of these is
// phonemized in pieces and the mark is put back between them, which is why
// `10:30` comes out `tˈɛn:θˈɜɹTi` rather than with a space.
var punctMarks = runeSet(`;:,.!?¡¿—…"«»“”(){}[]`)

// Open finds libespeak-ng and initialises it for one of the two Englishes.
//
// The search order is the environment first — `ESPEAK_LIB` or phonemizer's own
// `PHONEMIZER_ESPEAK_LIBRARY` — then the copy inside a local Python
// environment, then the usual system paths. The data directory is taken from
// `ESPEAK_DATA`, or from a sibling of the library, or left to espeak's own
// default.
func Open(british bool) (*Espeak, error) {
	lib, err := findEspeakLibrary()
	if err != nil {
		return nil, err
	}
	cname := C.CString(lib)
	defer C.free(unsafe.Pointer(cname))
	handle := C.dlopen(cname, C.RTLD_NOW|C.RTLD_LOCAL)
	if handle == nil {
		return nil, fmt.Errorf("g2p: dlopen %s: %s", lib, C.GoString(C.dlerror()))
	}
	e := &Espeak{handle: handle, british: british, Library: lib}
	for _, s := range []struct {
		name string
		dst  *unsafe.Pointer
	}{
		{"espeak_Initialize", &e.init},
		{"espeak_SetVoiceByName", &e.voice},
		{"espeak_TextToPhonemes", &e.ttp},
	} {
		cs := C.CString(s.name)
		sym := C.dlsym(handle, cs)
		C.free(unsafe.Pointer(cs))
		if sym == nil {
			C.dlclose(handle)
			return nil, fmt.Errorf("g2p: %s has no %s", lib, s.name)
		}
		*s.dst = sym
	}

	e.Data = espeakDataPath(lib)
	var cdata *C.char
	if e.Data != "" {
		cdata = C.CString(e.Data)
		defer C.free(unsafe.Pointer(cdata))
	}
	// A negative return is espeak's error; anything else is the sample rate,
	// which nothing here wants — no audio is produced.
	if rc := C.espeak_call_init(e.init, C.int(espeakSynch), 0, cdata, 0); rc < 0 {
		C.dlclose(handle)
		return nil, fmt.Errorf("g2p: espeak_Initialize(%q) returned %d", e.Data, int(rc))
	}
	name := "gmw/en-US"
	if british {
		name = "gmw/en-GB"
	}
	cv := C.CString(name)
	defer C.free(unsafe.Pointer(cv))
	if rc := C.espeak_call_voice(e.voice, cv); rc != 0 {
		C.dlclose(handle)
		return nil, fmt.Errorf("g2p: espeak_SetVoiceByName(%q) returned %d", name, int(rc))
	}
	return e, nil
}

// Close releases the library. espeak's own state is process-global and is left
// alone, since another Espeak may still be using it.
func (e *Espeak) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != nil {
		C.dlclose(e.handle)
		e.handle = nil
	}
}

// Raw is espeak's IPA for one piece of text, with `^` as the tie — which is
// what phonemizer produces and what `FromEspeak` expects.
//
// espeak consumes the text a clause at a time and advances the pointer, so
// this loops until the pointer comes back null. The pointer itself lives in C
// memory: cgo forbids handing C a Go pointer that might move, and the holder
// is exactly such a pointer.
func (e *Espeak) Raw(text string) string {
	if text == "" {
		return ""
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return ""
	}
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	holder := (*unsafe.Pointer)(C.malloc(C.size_t(unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(holder))
	*holder = unsafe.Pointer(ctext)

	mode := C.int(espeakIPA | espeakTieFlag | (espeakTie << 8))
	var parts []string
	for *holder != nil {
		out := C.espeak_call_ttp(e.ttp, holder, espeakUTF8, mode)
		if out == nil {
			break
		}
		if s := C.GoString(out); s != "" {
			parts = append(parts, s)
		}
	}
	return strings.ReplaceAll(strings.Join(parts, " "), string(espeakTie), "^")
}

// Phonemes is the whole fallback for one token: phonemizer's punctuation
// preservation, espeak, and misaki's rewrite.
//
// The punctuation split is the part that is phonemizer's rather than espeak's.
// Handed `10:30` whole, espeak reads it as two clauses and puts a space
// between them; phonemizer phonemizes the pieces separately and puts the mark
// back, which is what the model was trained on.
func (e *Espeak) Phonemes(text string) (string, int, bool) {
	var b strings.Builder
	var chunk strings.Builder
	flush := func() {
		if chunk.Len() == 0 {
			return
		}
		b.WriteString(FromEspeak(e.Raw(chunk.String()), e.british))
		chunk.Reset()
	}
	for _, r := range text {
		if punctMarks[r] {
			flush()
			b.WriteRune(r)
			continue
		}
		chunk.WriteRune(r)
	}
	flush()
	out := strings.TrimSpace(b.String())
	if out == "" {
		return "", 0, false
	}
	return out, RatingEspeak, true
}

func findEspeakLibrary() (string, error) {
	var tried []string
	for _, env := range []string{"ESPEAK_LIB", "PHONEMIZER_ESPEAK_LIBRARY"} {
		if p := os.Getenv(env); p != "" {
			if _, err := os.Stat(p); err == nil {
				return p, nil
			}
			tried = append(tried, p+" (from $"+env+")")
		}
	}
	var candidates []string
	// The copy inside a local Python environment, which is where misaki's own
	// espeak comes from and the only one on this machine.
	for _, glob := range []string{
		".venv/lib/python*/site-packages/espeakng_loader/libespeak-ng.so*",
		"../.venv/lib/python*/site-packages/espeakng_loader/libespeak-ng.so*",
	} {
		m, _ := filepath.Glob(glob)
		candidates = append(candidates, m...)
	}
	candidates = append(candidates,
		"/usr/lib/libespeak-ng.so.1", "/usr/lib/libespeak-ng.so",
		"/usr/lib64/libespeak-ng.so.1", "/usr/lib64/libespeak-ng.so",
		"/usr/local/lib/libespeak-ng.so.1", "/usr/local/lib/libespeak-ng.so",
		"/opt/homebrew/lib/libespeak-ng.dylib",
	)
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	tried = append(tried, candidates...)
	return "", fmt.Errorf("g2p: no libespeak-ng found; set $ESPEAK_LIB (looked in %s)",
		strings.Join(tried, ", "))
}

// espeakDataPath is the directory *containing* `espeak-ng-data`, which is what
// espeak_Initialize wants.
func espeakDataPath(lib string) string {
	if p := os.Getenv("ESPEAK_DATA"); p != "" {
		return p
	}
	dir := filepath.Dir(lib)
	for _, d := range []string{dir, filepath.Join(dir, ".."), "/usr/share", "/usr/local/share"} {
		if fi, err := os.Stat(filepath.Join(d, "espeak-ng-data")); err == nil && fi.IsDir() {
			return d
		}
	}
	return ""
}
