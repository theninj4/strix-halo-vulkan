// The dense 4.5-bit bank's record, as the kernels read it: LLM.md L8c-4.
//
// A record is sixteen bytes — `d` and `dmin` as halves, then the twelve bytes
// ggml packs eight 6-bit (scale, min) pairs into — and it covers one output
// column of one 256-element super-block. `llm/bank_q4.go` writes them; the
// GEMM (`-DQ4B`) and the GEMV (`-DQ4B`) read them.
//
// `get_scale_min_k4` below is ggml's, and the same function
// `llm_moe_gemm.comp` and `llm_moe_gemv.comp` carry for the *expert* bank.
// It is a file of its own here because those two read the checkpoint's own
// row-major Q4_K blocks where these two read §2.8 fragment tiles, so the
// addressing around it has nothing in common and only the twelve bytes do.

uint q4kByte(uvec3 s, uint i) {
    uint w = (i < 4u) ? s.x : ((i < 8u) ? s.y : s.z);
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

void q4kScaleMin(uint j, uvec3 s, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = q4kByte(s, j) & 63u;
        mn = q4kByte(s, j + 4u) & 63u;
    } else {
        uint a = q4kByte(s, j + 4u);
        sc = (a & 0xFu) | ((q4kByte(s, j - 4u) >> 6) << 4);
        mn = (a >> 4) | ((q4kByte(s, j) >> 6) << 4);
    }
}
