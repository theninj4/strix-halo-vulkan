// eval_dump_mtmd is eval_dump with an image in the prompt: the oracle for
// LLM-VISION.md V6.
//
// llama.cpp runs a prompt holding one image through mtmd (its vision tower,
// its preprocessing, its M-RoPE positions) and the language model, and this
// writes what the Go graph needs to replay exactly the same pass. Three
// files, and the LLM's own tensors:
//
//	tokens.bin      the EVTK token file eval_dump writes, with the image's
//	                cells as the image-pad id (248056): the ids the graph's
//	                PLE hashes there, and what llama.cpp stores in those
//	                cells too (llama-kv-cache.cpp, ple_image_token_id)
//	image.txt       "at nx ny n_tokens n_embd": where the image starts in
//	                the token stream, and its merged grid
//	image_embd.bin  mtmd's own embedding rows, [n_tokens][n_embd] f32. The
//	                Go side feeds *these* to the graph, so the comparison is
//	                the language model's and not the two towers' (whose
//	                preprocessing differs: stb_image and llama.cpp's resize
//	                against HF's). How far our tower is from these rows is
//	                measured separately
//	NNNN_*.bin      the eval callback's tensors, one set per llama_decode:
//	                mtmd decodes the text before the image, the image, and
//	                the text after it as three calls, so a tensor name
//	                recurs, and the reader concatenates the occurrences in
//	                order
//
//	L=/home/kube/repos/llama.cpp
//	gcc -O2 -o /tmp/eval_dump_mtmd reference/eval_dump_mtmd.c \
//	    -I$L/include -I$L/ggml/include -I$L/tools/mtmd -L$L/build/bin \
//	    -lllama -lggml-base -lmtmd -Wl,-rpath,$L/build/bin
//	/tmp/eval_dump_mtmd -m MODEL.gguf -mm MMPROJ.gguf -i IMAGE -o DIR \
//	    -p 'text <__media__> text' -n 'regex' [-c ctx]
//
// The marker is mtmd's default, `<__media__>`, and mtmd itself puts
// <|vision_start|> and <|vision_end|> around the image, so the prompt names
// only where the picture goes.

#define main eval_dump_main
#include "eval_dump.c"
#undef main

#include "mtmd.h"
#include "mtmd-helper.h"

int main(int argc, char **argv) {
    const char *model = NULL, *mmproj = NULL, *image = NULL, *prompt = NULL, *pattern = NULL;
    int n_ctx = 4096;
    snprintf(outdir, sizeof(outdir), "%s", "reference/out/llmvision_eval");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)       { model  = argv[++i]; }
        else if (!strcmp(argv[i], "-mm") && i + 1 < argc) { mmproj = argv[++i]; }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)  { image  = argv[++i]; }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc)  { prompt = argv[++i]; }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)  { pattern = argv[++i]; }
        else if (!strcmp(argv[i], "-c") && i + 1 < argc)  { n_ctx  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc)  { snprintf(outdir, sizeof(outdir), "%s", argv[++i]); }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model || !mmproj || !image || !prompt) {
        fprintf(stderr, "usage: %s -m model.gguf -mm mmproj.gguf -i image -p prompt [-o dir] [-n regex] [-c ctx]\n", argv[0]);
        return 2;
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
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.cb_eval = dump_cb;
    struct llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    struct mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = true;
    mcp.warmup = false;
    mtmd_context *mctx = mtmd_init_from_file(mmproj, m, mcp);
    if (!mctx) { fprintf(stderr, "failed to load %s\n", mmproj); return 1; }

    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mctx, image, false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { fprintf(stderr, "failed to read %s\n", image); return 1; }
    mtmd_bitmap *bmp = bw.bitmap;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    struct mtmd_input_text text = { prompt, strlen(prompt), true, true };
    const mtmd_bitmap *bitmaps[1] = { bmp };
    if (mtmd_tokenize(mctx, chunks, &text, bitmaps, 1) != 0) { fprintf(stderr, "mtmd_tokenize failed\n"); return 1; }

    // The token stream as the Go graph will run it, and the image's place.
    const int32_t pad = 248056;
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    llama_token *toks = (llama_token *) malloc((size_t) n_ctx * sizeof(llama_token));
    int32_t nt = 0, at = -1, nx = 0, ny = 0, n_img = 0;
    for (size_t c = 0; c < n_chunks; c++) {
        const mtmd_input_chunk *ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token *t = mtmd_input_chunk_get_tokens_text(ch, &n);
            for (size_t k = 0; k < n && nt < n_ctx; k++) { toks[nt++] = t[k]; }
            continue;
        }
        const mtmd_image_tokens *it = mtmd_input_chunk_get_tokens_image(ch);
        at = nt;
        nx = (int32_t) mtmd_image_tokens_get_nx(it);
        ny = (int32_t) mtmd_image_tokens_get_ny(it);
        n_img = (int32_t) mtmd_image_tokens_get_n_tokens(it);
        for (int32_t k = 0; k < n_img && nt < n_ctx; k++) { toks[nt++] = pad; }

        // Its embedding rows: the tower's output, exactly as the decode
        // below consumes it.
        if (mtmd_encode_chunk(mctx, ch) != 0) { fprintf(stderr, "mtmd_encode_chunk failed\n"); return 1; }
        const float *embd = mtmd_get_output_embd(mctx);
        int32_t n_embd = llama_model_n_embd(m);
        char path[2048];
        snprintf(path, sizeof(path), "%s/image_embd.bin", outdir);
        FILE *ef = fopen(path, "wb");
        fwrite(embd, sizeof(float), (size_t) n_img * (size_t) n_embd, ef);
        fclose(ef);
        snprintf(path, sizeof(path), "%s/image.txt", outdir);
        FILE *mf = fopen(path, "w");
        fprintf(mf, "%d %d %d %d %d\n", at, nx, ny, n_img, n_embd);
        fclose(mf);
    }
    if (at < 0) { fprintf(stderr, "the prompt holds no image\n"); return 1; }
    fprintf(stderr, "eval_dump_mtmd: %d tokens, image at %d, %dx%d = %d tokens\n", nt, at, nx, ny, n_img);

    char path[2048];
    snprintf(path, sizeof(path), "%s/tokens.bin", outdir);
    FILE *tf = fopen(path, "wb");
    fwrite("EVTK", 1, 4, tf);
    put_u32(tf, (uint32_t) nt);
    for (int i = 0; i < nt; i++) { put_u32(tf, (uint32_t) toks[i]); }
    put_str(tf, prompt);
    fclose(tf);

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(mctx, ctx, chunks, 0, 0, n_ctx, true, &n_past) != 0) {
        fprintf(stderr, "eval failed\n"); return 1;
    }
    fprintf(stderr, "eval_dump_mtmd: n_past %d after %d cells; wrote %d tensors, %.1f MB to %s\n",
            n_past, nt, n_written, bytes_written / 1048576.0, outdir);
    mtmd_input_chunks_free(chunks);
    mtmd_free(mctx);
    llama_backend_free();
    return 0;
}
