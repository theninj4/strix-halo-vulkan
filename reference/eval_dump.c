// eval_dump writes llama.cpp's own intermediate activations to disk, full
// tensors rather than the 3-per-axis corners `llama-eval-callback` prints, so
// that the Go implementation of a layer can be checked value for value
// instead of against a sum.
//
// It is the oracle for LLM.md's L2 onwards, the same way reference/
// dequant_ref.c is the oracle for gguf/dequant.go: it links against the
// already-built libllama and registers the *same* eval callback the scheduler
// hands every node, so what it records is the reference implementation's own
// tensors and not a re-reading of its source.
//
//	L=/home/kube/repos/llama.cpp
//	gcc -O2 -o /tmp/eval_dump reference/eval_dump.c \
//	    -I$L/include -I$L/ggml/include -L$L/build/bin \
//	    -lllama -lggml-base -Wl,-rpath,$L/build/bin
//	/tmp/eval_dump -m MODEL.gguf -o DIR -p 'hello world' -n 'regex'
//	/tmp/eval_dump -m MODEL.gguf -o DIR -f prompt.txt -nt 4096 -c 4096 -n 'regex'
//
// `-f` reads the prompt from a file and `-nt` truncates the token list to an
// exact length, which is how L4's 4 k fixture is cut: the QSA selection only
// starts excluding cells past `top_k + ratio - 1` = 2051, so the dump that
// tests it has to be longer than that and a round 4096 makes the block
// arithmetic legible.
//
// Only tensors whose name matches the POSIX extended regex are written, which
// matters: an unfiltered pass over this model is 1224 tensors a layer-stack
// and tens of gigabytes. Quantised nodes are skipped (they are weights, which
// the Go side reads from the GGUF itself).
//
// One file per *write*, named `NNNN_name.bin` with '/' -> '%'. The sequence
// number matters: `build_hc_mix` runs twice a layer, so `hc_norm-3` is two
// different tensors in one pass, and a name alone would have the second
// silently overwrite the first. NNNN is also the graph order, which is what a
// reader wants when it is walking a layer. Format, little-endian:
//
//	magic  "EVDP"            4 bytes
//	type   u32               ggml_type of the payload (0 f32, 1 f16, 30 bf16)
//	ne     u64 x 4           ggml shape, ne[0] fastest
//	nb     u64 x 4           ggml strides in bytes, as the tensor was laid out
//	nbytes u64               payload length
//	name   u32 len, bytes    the ggml tensor name
//	op     u32 len, bytes    ggml_op_desc
//	payload
//
// Strides are recorded because a dumped view need not be contiguous; the
// reader is expected to walk nb rather than assume a packed layout.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <regex.h>
#include <sys/stat.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

static regex_t  filter;
static int      have_filter = 0;
static char     outdir[1024];
static int      n_written = 0;
static uint64_t bytes_written = 0;

static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void put_str(FILE *f, const char *s) {
    uint32_t n = (uint32_t) strlen(s);
    put_u32(f, n);
    fwrite(s, 1, n, f);
}

static bool dump_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    (void) user_data;
    if (ask) {
        // The scheduler asks before it computes; answering false here keeps
        // the node from being split out of its backend graph unnecessarily.
        return !have_filter || regexec(&filter, t->name, 0, NULL, 0) == 0;
    }
    if (have_filter && regexec(&filter, t->name, 0, NULL, 0) != 0) { return true; }
    if (ggml_is_quantized(t->type)) { return true; }

    const size_t nbytes = ggml_nbytes(t);
    uint8_t *buf = (uint8_t *) malloc(nbytes);
    if (!buf) { fprintf(stderr, "eval_dump: out of memory for %s\n", t->name); return true; }
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(buf, t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, buf, 0, nbytes);
    }

    char safe[512];
    snprintf(safe, sizeof(safe), "%s", t->name);
    for (char *p = safe; *p; p++) { if (*p == '/') { *p = '%'; } }

    char path[2048];
    snprintf(path, sizeof(path), "%s/%04d_%s.bin", outdir, n_written, safe);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); free(buf); return true; }

    fwrite("EVDP", 1, 4, f);
    put_u32(f, (uint32_t) t->type);
    for (int i = 0; i < GGML_MAX_DIMS; i++) { put_u64(f, (uint64_t) t->ne[i]); }
    for (int i = 0; i < GGML_MAX_DIMS; i++) { put_u64(f, (uint64_t) t->nb[i]); }
    put_u64(f, (uint64_t) nbytes);
    put_str(f, t->name);
    put_str(f, ggml_op_desc(t));
    fwrite(buf, 1, nbytes, f);
    fclose(f);
    free(buf);

    bytes_written += nbytes;
    n_written++;
    fprintf(stderr, "eval_dump: %04d %-36s %s [%lld,%lld,%lld,%lld] %zu B\n",
            n_written - 1, t->name, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3], nbytes);
    return true;
}

// read_file slurps a prompt file. The 4 k fixture is ~20 KB of wikitext, which
// is not something to paste onto a command line.
static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *) malloc((size_t) n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) { free(buf); fclose(f); return NULL; }
    buf[n] = 0;
    fclose(f);
    *len = (size_t) n;
    return buf;
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt = "hello world", *pattern = NULL, *promptfile = NULL;
    int n_ctx = 512, n_ubatch = 0, n_trunc = 0;
    size_t prompt_len = 0;
    snprintf(outdir, sizeof(outdir), "%s", "reference/out/eval");

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)      { model   = argv[++i]; }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) { prompt  = argv[++i]; }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) { pattern = argv[++i]; }
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) { promptfile = argv[++i]; }
        else if (!strcmp(argv[i], "-nt") && i + 1 < argc){ n_trunc = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) { n_ctx   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-ub") && i + 1 < argc){ n_ubatch= atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) { snprintf(outdir, sizeof(outdir), "%s", argv[++i]); }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model) { fprintf(stderr, "usage: %s -m model.gguf [-o dir] [-p prompt | -f file] [-nt tokens] [-n regex] [-c ctx] [-ub ubatch]\n", argv[0]); return 2; }
    if (promptfile) {
        prompt = read_file(promptfile, &prompt_len);
        if (!prompt) { return 1; }
    } else {
        prompt_len = strlen(prompt);
    }
    if (pattern) {
        if (regcomp(&filter, pattern, REG_EXTENDED | REG_NOSUB) != 0) {
            fprintf(stderr, "bad regex: %s\n", pattern); return 2;
        }
        have_filter = 1;
    }
    mkdir(outdir, 0755);

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    struct llama_model *m = llama_model_load_from_file(model, mp);
    if (!m) { fprintf(stderr, "failed to load %s\n", model); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = n_ctx;
    cp.n_batch  = n_ctx;
    cp.n_ubatch = n_ubatch ? (uint32_t) n_ubatch : (uint32_t) n_ctx;
    cp.cb_eval  = dump_cb;
    cp.cb_eval_user_data = NULL;
    struct llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(m);
    // The cap is the prompt's own length in bytes: a token is at least one
    // byte, so this cannot be short, and a 4 k fixture will not fit a fixed
    // 4096-entry array once BOS and the chat specials are counted.
    const int32_t cap = (int32_t) prompt_len + 8;
    llama_token *toks = (llama_token *) malloc((size_t) cap * sizeof(llama_token));
    if (!toks) { fprintf(stderr, "out of memory for %d tokens\n", cap); return 1; }
    int32_t nt = llama_tokenize(vocab, prompt, (int32_t) prompt_len, toks, cap, true, true);
    if (nt <= 0) { fprintf(stderr, "tokenize failed: %d\n", nt); return 1; }
    // -nt cuts the prompt to an exact token count. The fixture's length is
    // arithmetic the tests read back (block counts, the top-k width, where
    // the selection starts to bite), so it wants to be a chosen number and
    // not whatever a paragraph boundary happened to give.
    if (n_trunc > 0) {
        if (nt < n_trunc) { fprintf(stderr, "prompt is %d tokens, -nt asked for %d\n", nt, n_trunc); return 1; }
        nt = n_trunc;
    }
    if (nt > (int32_t) n_ctx) { fprintf(stderr, "prompt is %d tokens, context is %d\n", nt, n_ctx); return 1; }
    fprintf(stderr, "eval_dump: %d tokens", nt);
    for (int i = 0; i < nt && i < 32; i++) { fprintf(stderr, " %d", toks[i]); }
    fprintf(stderr, "%s\n", nt > 32 ? " ..." : "");

    // The token ids are the other half of the fixture: without them the Go
    // side cannot reproduce the same forward pass.
    char path[2048];
    snprintf(path, sizeof(path), "%s/tokens.bin", outdir);
    FILE *tf = fopen(path, "wb");
    if (tf) {
        fwrite("EVTK", 1, 4, tf);
        put_u32(tf, (uint32_t) nt);
        for (int i = 0; i < nt; i++) { put_u32(tf, (uint32_t) toks[i]); }
        // The ids are the fixture; the text is a convenience, and after -nt
        // it no longer spells them, so it is dropped rather than made to lie.
        put_str(tf, n_trunc > 0 ? "" : prompt);
        fclose(tf);
    }

    if (llama_decode(ctx, llama_batch_get_one(toks, nt)) != 0) {
        fprintf(stderr, "decode failed\n"); return 1;
    }

    fprintf(stderr, "eval_dump: wrote %d tensors, %.1f MB to %s\n",
            n_written, bytes_written / 1048576.0, outdir);
    llama_backend_free();
    return 0;
}
