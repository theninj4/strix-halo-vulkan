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

// **The ten-group record**: LLM.md L8c-6, and the one place this bank is not
// ggml's.
//
// `get_scale_min_k4` above *is* eight groups — four pairs carried whole and
// four whose high bits are stolen from the first four's spare ones — so there
// is no ten-group spelling of it. The hyper-connection block's up projection
// reads the low-rank space and is 320 wide, which is ten groups of 32 and one
// super-block for the whole row, so its record is twenty bytes rather than
// sixteen: the fp16 pair, then twelve bits a group at bit 12*j of a
// little-endian bit stream, scale in the low six and min in the high six.
//
// The decode is a shift and a mask where ggml's is a branch. j*12 crosses a
// word boundary only at j = 2 and j = 5, and in both the next word exists, so
// there is no guard on the high half beyond the shift that makes it zero.
void q4kScaleMin12(uint j, uvec4 s, out uint sc, out uint mn) {
    uint b = j * 12u;
    uint w = b >> 5, o = b & 31u;
    uint v = s[w] >> o;
    if (o > 20u) {
        v |= s[w + 1u] << (32u - o);
    }
    sc = v & 63u;
    mn = (v >> 6) & 63u;
}

// **P3a's fifth bit.** `-DQ5B` (always beside `-DQ4B`) adds ggml's `qh` to
// the bank: the record above does not move — ggml's own Q5_K carries Q4_K's
// record — and neither does the nibble tiling. What is added is one plane of
// one bit a weight, sitting **between** the tiles and the records, so both
// following bases stay derivable from `gemmN * gemmK` and neither needs a
// push field (64 uints is this device's whole push range, full since L5b).
//
// The plane is one 32-byte tile per 128-byte nibble tile, in the same order,
// and **byte i of it is the top bit of each of the eight levels in word i of
// the nibble tile**, bit e for the level at k offset e inside that word. So
// the plane byte index *is* the nibble word index, and every kernel reads it
// with the address arithmetic it already had: `QK_HIGH_BYTE` takes that word
// index and the plane's word base and returns the eight bits.
#ifdef Q5B
#define QK_HIGH_BYTES(NK) ((NK) >> 3)
#define QK_HIGH_BYTE(H0, WI) ((w8[(H0) + ((WI) >> 2)] >> (((WI) & 3u) * 8u)) & 0xFFu)
// The level of the nibble pair in byte `t` of a word: the low nibble carries
// bit 2t of the plane byte and the high one bit 2t+1.
#define QK_LEVEL_LO(BY, HB, T) (((BY) & 0xFu) | ((((HB) >> ((T) * 2u)) & 1u) << 4))
#define QK_LEVEL_HI(BY, HB, T) (((BY) >> 4) | ((((HB) >> ((T) * 2u + 1u)) & 1u) << 4))
#else
#define QK_HIGH_BYTES(NK) 0u
#define QK_HIGH_BYTE(H0, WI) 0u
#define QK_LEVEL_LO(BY, HB, T) ((BY) & 0xFu)
#define QK_LEVEL_HI(BY, HB, T) ((BY) >> 4)
#endif
