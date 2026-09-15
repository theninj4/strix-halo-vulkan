// dequant_ref dumps llama.cpp's own dequantisation of a fixed set of blocks,
// so the Go paths in gguf/dequant.go can be checked against the reference
// implementation rather than against a re-reading of the same source.
//
// It is the oracle half of gguf's TestDequantMatchesGGML. The Go test
// generates the input blocks (deterministically, with the fp16 scale fields
// constrained to ordinary values so that no comparison lands on a NaN), and
// this reads them back through ggml_get_type_traits(type)->to_float — the
// same function every llama.cpp CPU path uses.
//
//	L=/home/kube/repos/llama.cpp
//	gcc -O2 -o /tmp/dequant_ref reference/dequant_ref.c \
//	    -I$L/ggml/include -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin
//	go test ./gguf/ -run TestDequantMatchesGGML -updatedequant   # writes the input
//	/tmp/dequant_ref gguf/testdata/dequant_in.bin gguf/testdata/dequant_ref.bin
//
// Both files are committed: the test is then a plain comparison with no
// toolchain requirement, and regenerating them is the only step that needs
// llama.cpp present.
//
// Record format, both files:
//	u32 ggml type, u32 element count, u64 payload bytes, payload
// where the input's payload is the packed block bytes and the output's is
// that many float32s.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ggml.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]); return 2; }
    FILE *in = fopen(argv[1], "rb");
    if (!in) { perror(argv[1]); return 1; }
    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 1; }

    uint32_t type, n;
    uint64_t nbytes;
    int records = 0;
    while (fread(&type, 4, 1, in) == 1) {
        if (fread(&n, 4, 1, in) != 1 || fread(&nbytes, 8, 1, in) != 1) {
            fprintf(stderr, "truncated record header\n"); return 1;
        }
        void *buf = malloc(nbytes);
        if (fread(buf, 1, nbytes, in) != nbytes) { fprintf(stderr, "truncated payload\n"); return 1; }

        const struct ggml_type_traits *tr = ggml_get_type_traits((enum ggml_type) type);
        float *y = malloc((size_t) n * sizeof(float));
        if (tr->to_float) {
            tr->to_float(buf, y, (int64_t) n);
        } else if ((enum ggml_type) type == GGML_TYPE_F32) {
            // F32 is already float; ggml leaves its to_float null rather than
            // pointing it at a memcpy.
            memcpy(y, buf, (size_t) n * sizeof(float));
        } else {
            fprintf(stderr, "type %u (%s) has no to_float\n", type, tr->type_name);
            return 1;
        }

        uint64_t outbytes = (uint64_t) n * sizeof(float);
        fwrite(&type, 4, 1, out);
        fwrite(&n, 4, 1, out);
        fwrite(&outbytes, 8, 1, out);
        fwrite(y, 1, outbytes, out);
        fprintf(stderr, "%-8s %6u elements from %6llu bytes\n", tr->type_name, n, (unsigned long long) nbytes);

        free(buf); free(y);
        records++;
    }
    fclose(in); fclose(out);
    fprintf(stderr, "%d records\n", records);
    return 0;
}
