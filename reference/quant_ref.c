// quant_ref runs llama.cpp's own quantiser over rows of f32 and dumps what it
// produces, so that llm/sim.go's port of `make_qx_quants` can be checked
// against the reference implementation rather than against a re-reading of the
// same source (LLM.md L8c-2).
//
// It is the oracle half of llm's TestQuantSimMatchesGGML. The Go test writes
// real weight rows out of the checkpoint — and, for the calibrated arm, the
// matching row of unsloth's published imatrix — and this quantises them with
// `ggml_quantize_chunk`, which is the same entry point `llama-quantize` uses,
// then dequantises straight back through `ggml_get_type_traits(t)->to_float`.
// The Go side compares against its own round trip.
//
// Passing an imatrix or not is the whole point: `quantize_row_q4_0_impl`
// short-circuits to `quantize_row_q4_0_ref` when it is null, so one binary
// produces both the uncalibrated and the calibrated reference.
//
//	L=/home/kube/repos/llama.cpp
//	gcc -O2 -o /tmp/quant_ref reference/quant_ref.c \
//	    -I$L/ggml/include -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin
//	go test ./llm/ -run TestQuantSimMatchesGGML -updatequant   # writes the input
//	/tmp/quant_ref llm/testdata/quant_in.bin llm/testdata/quant_ref.bin
//
// Both files are committed, so the test is then a plain comparison with no
// toolchain requirement, on dequant_ref.c's precedent.
//
// Input record format, repeated until EOF:
//	u32 ggml type, u32 rows, u32 n_per_row, u32 have_imatrix,
//	rows*n_per_row f32 weights, then n_per_row f32 importances if have_imatrix
// Output record format, one per input record:
//	u32 ggml type, u32 rows, u32 n_per_row, u32 have_imatrix,
//	rows*n_per_row f32 — the quantised values dequantised back

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ggml.h"

static void die(const char *m) {
    fprintf(stderr, "quant_ref: %s\n", m);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in.bin> <out.bin>\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) die("cannot open input");
    FILE *out = fopen(argv[2], "wb");
    if (!out) die("cannot open output");

    for (;;) {
        uint32_t hdr[4];
        size_t got = fread(hdr, sizeof(uint32_t), 4, in);
        if (got == 0) break;
        if (got != 4) die("short header");
        const enum ggml_type type = (enum ggml_type) hdr[0];
        const int64_t rows = hdr[1], n_per_row = hdr[2];
        const int have_im = hdr[3];
        const int64_t n = rows * n_per_row;

        float *src = malloc(n * sizeof(float));
        if (!src) die("oom");
        if ((int64_t) fread(src, sizeof(float), n, in) != n) die("short weights");

        float *im = NULL;
        if (have_im) {
            im = malloc(n_per_row * sizeof(float));
            if (!im) die("oom");
            if ((int64_t) fread(im, sizeof(float), n_per_row, in) != n_per_row) die("short imatrix");
        }

        // The quantised bytes, then straight back to floats. This is the
        // same pair of calls llama-quantize makes per tensor chunk.
        const size_t qbytes = ggml_row_size(type, n_per_row) * rows;
        void *q = malloc(qbytes);
        if (!q) die("oom");
        ggml_quantize_chunk(type, src, q, 0, rows, n_per_row, im);

        float *back = malloc(n * sizeof(float));
        if (!back) die("oom");
        const struct ggml_type_traits *tr = ggml_get_type_traits(type);
        if (!tr->to_float) die("type has no to_float");
        tr->to_float(q, back, n);

        if (fwrite(hdr, sizeof(uint32_t), 4, out) != 4) die("short write");
        if ((int64_t) fwrite(back, sizeof(float), n, out) != n) die("short write");

        free(src); free(im); free(q); free(back);
    }
    fclose(in);
    fclose(out);
    return 0;
}
