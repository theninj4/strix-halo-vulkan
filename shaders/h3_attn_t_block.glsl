// One key block of h3_attn_t.comp, included twice when TAIL_SPLIT is set:
// MASKED is 0 for a whole block (cnt == BN, known at compile time, so no
// tail arithmetic is compiled into it) and 1 for the last one. Expects
// base, cnt, fragQ/qs, o, runMax, runSum, hw in scope.
//
// L0_LOADS=1 (a screening control, wrong by construction): every K and V
// fragment load reads the first key block, so the loads are issued and
// waited for as before but every one is an L0 hit (KERNELS.md G1's control).
#if L0_LOADS
#define KV_BLOCK 0u
#else
#define KV_BLOCK (base / TILE)
#endif

        KV_BEFORE_S

        // ---- S^T = K . Q^T.
        ACC s[QT][KTIL];
        [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
            [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                s[qq][kk] = ACC(0.0);
            }
        }
        [[unroll]] for (uint kt = 0u; kt < HDT; ++kt) {
            FRAG_A fragK[KTIL];
            [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
#if KVLDS
                coopMatLoad(fragK[kk], lsK, (kk * TILE) * K_LPITCH + kt * TILE,
                    K_LPITCH, gl_CooperativeMatrixLayoutRowMajor);
#else
                coopMatLoad(fragK[kk], hact, kBase + ((KV_BLOCK + kk) * HDT + kt) * (TILE * TILE),
                    TILE, gl_CooperativeMatrixLayoutRowMajor);
#endif
            }
            [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
#if QLDS
                FRAG_B fq;
                coopMatLoad(fq, qs, (qq * HDT + kt) * Q_LTILE, Q_LPITCH / 2u,
                    gl_CooperativeMatrixLayoutColumnMajor);
#elif QREG
                FRAG_B fq = fragQ[qq][kt];
#else
                FRAG_B fq;
                coopMatLoad(fq, hact, qBase + (qq * HDT + kt) * (TILE * TILE),
                    TILE, gl_CooperativeMatrixLayoutColumnMajor);
#endif
                [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                    s[qq][kk] = coopMatMulAdd(fragK[kk], fq, s[qq][kk]);
                }
            }
        }

#if !KV_BSOFT
        KV_AFTER_S
#endif

        // ---- The softmax, in registers: a lane's elements of s[qq][kk] are
        // its query's scores against keys kk*16 + 2e + hw.
        FRAG_B fragP[QT][KTIL];
#if PV_ORDER
        float corrq[QT], mnq[QT];
#endif
        [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
#if GUT_SOFTMAX
            // A3's floor: P^T is S^T narrowed, no max, exp, correction or sum.
            [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                    f16vec2 h = f16vec2(s[qq][kk][2u * i], s[qq][kk][2u * i + 1u]);
#if GUT_XCHG
                    fragP[qq][kk][4u * i] = h.x;
                    fragP[qq][kk][4u * i + 1u] = h.y;
                    fragP[qq][kk][4u * i + 2u] = h.x;
                    fragP[qq][kk][4u * i + 3u] = h.y;
#else
                    uint wi = packFloat2x16(h);
                    f16vec2 mine = unpackFloat2x16(wi);
                    f16vec2 theirs = unpackFloat2x16(subgroupShuffleXor(wi, 16u));
                    f16vec2 even = hw == 0u ? mine : theirs;
                    f16vec2 odd = hw == 0u ? theirs : mine;
                    fragP[qq][kk][4u * i] = even.x;
                    fragP[qq][kk][4u * i + 1u] = odd.x;
                    fragP[qq][kk][4u * i + 2u] = even.y;
                    fragP[qq][kk][4u * i + 3u] = odd.y;
#endif
                }
            }
#else
            float mx = -1.0 / 0.0;
#if GUT_MAX
            mx = 0.0;
#else
            if (MASKED == 0 || cnt == BN) {
#if MAX_TREE
                // A balanced tree: max is exact, so the order is free, and
                // a chain of 2*KTIL dependent max3 is a latency the other
                // waves do not always cover (research/3.8).
                float mk[KTIL];
                [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                    mk[kk] = max(max(max(s[qq][kk][0], s[qq][kk][1]), max(s[qq][kk][2], s[qq][kk][3])),
                                 max(max(s[qq][kk][4], s[qq][kk][5]), max(s[qq][kk][6], s[qq][kk][7])));
                }
#if KTIL == 4
                mx = max(max(mk[0], mk[1]), max(mk[2], mk[3]));
#elif KTIL == 2
                mx = max(mk[0], mk[1]);
#elif KTIL == 8
                mx = max(max(max(mk[0], mk[1]), max(mk[2], mk[3])), max(max(mk[4], mk[5]), max(mk[6], mk[7])));
#else
                [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                    mx = max(mx, mk[kk]);
                }
#endif
#else
                [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                    [[unroll]] for (uint e = 0u; e < 8u; ++e) {
                        mx = max(mx, s[qq][kk][e]);
                    }
                }
#endif
            } else {
                [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                    [[unroll]] for (uint e = 0u; e < 8u; ++e) {
                        if (kk * TILE + 2u * e + hw < cnt) {
                            mx = max(mx, s[qq][kk][e]);
                        }
                    }
                }
            }
            mx = max(mx, subgroupShuffleXor(mx, 16u));
#endif
            float mo = runMax[qq];
            float mn = max(mo, mx);
            runMax[qq] = mn;
            // exp2(-inf - m) is 0: the first block's correction, no special case.
            float corr = exp2(mo - mn);
#if !GUT_CORR
#if LAZY
            // Exact laziness: when no lane's max moved, corr is exp2(0) = 1
            // and the rescale is the identity, so it is skipped (uniformly).
            if (subgroupAny(mo != mn)) {
#endif
            [[unroll]] for (uint j = 0u; j < HDT; ++j) {
                [[unroll]] for (uint e = 0u; e < 8u; ++e) {
                    o[qq][j][e] *= corr;
                }
            }
#if LAZY
            }
#endif
#endif
            float sum = 0.0;
#if PV_ORDER
            corrq[qq] = corr;
            mnq[qq] = mn;
        }
        // The P^T tiles a key tile at a time, each one's MMAs right behind
        // it, so S^T dies as P^T is born and the peak register set is
        // smaller: o[qq][j]'s MMAs come in the same kk order as before.
        float sumq[QT];
        [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
            sumq[qq] = 0.0;
        }
        [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
            FRAG_B fragPk[QT];
            [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
                float mn = mnq[qq];
                float sum = sumq[qq];
#define fragP_qq_kk fragPk[qq]
#else
            [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
#define fragP_qq_kk fragP[qq][kk]
#endif
                // P^T narrowed to fp16, two keys a word: w[i] holds this
                // lane's keys 4i + hw and 4i + 2 + hw.
                uint w[4];
                [[unroll]] for (uint i = 0u; i < 4u; ++i) {
#if GUT_EXP
                    float p0 = s[qq][kk][2u * i] - mn;
                    float p1 = s[qq][kk][2u * i + 1u] - mn;
#else
                    float p0 = exp2(s[qq][kk][2u * i] - mn);
                    float p1 = exp2(s[qq][kk][2u * i + 1u] - mn);
#endif
                    if (MASKED != 0 && cnt < BN) {
                        uint k0 = kk * TILE + 4u * i + hw;
                        p0 = k0 < cnt ? p0 : 0.0;
                        p1 = k0 + 2u < cnt ? p1 : 0.0;
                    }
                    f16vec2 h = f16vec2(p0, p1);
                    sum += float(h.x) + float(h.y);
                    w[i] = packFloat2x16(h);
                }
                // B wants keys 0..15 of this lane's query in order. The other
                // half-wave holds the other parity; swap and interleave.
                [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                    f16vec2 mine = unpackFloat2x16(w[i]);
                    f16vec2 theirs = unpackFloat2x16(subgroupShuffleXor(w[i], 16u));
                    f16vec2 even = hw == 0u ? mine : theirs;   // keys 4i, 4i+2
                    f16vec2 odd = hw == 0u ? theirs : mine;    // keys 4i+1, 4i+3
                    fragP_qq_kk[4u * i] = even.x;
                    fragP_qq_kk[4u * i + 1u] = odd.x;
                    fragP_qq_kk[4u * i + 2u] = even.y;
                    fragP_qq_kk[4u * i + 3u] = odd.y;
                }
#undef fragP_qq_kk
#if PV_ORDER
                sumq[qq] = sum;
            }
            [[unroll]] for (uint j = 0u; j < HDT; ++j) {
                FRAG_A fragV;
                coopMatLoad(fragV, hact, vBase + ((KV_BLOCK + kk) * HDT + j) * (TILE * TILE),
                    TILE, gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
                    o[qq][j] = coopMatMulAdd(fragV, fragPk[qq], o[qq][j]);
                }
            }
        }
        [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
            runSum[qq] = runSum[qq] * corrq[qq] + sumq[qq];
        }
#else
            }
            runSum[qq] = runSum[qq] * corr + sum;
#endif
#endif
#if !PV_ORDER || GUT_SOFTMAX
        }
#if KV_BSOFT
        // KV_BSOFT: the barrier that ends the K reads behind the softmax.
        KV_AFTER_S
#endif

        // ---- O^T += V^T . P^T.
        [[unroll]] for (uint j = 0u; j < HDT; ++j) {
            FRAG_A fragV[KTIL];
            [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
#if KVLDS
                coopMatLoad(fragV[kk], lsV, (j * TILE) * V_LPITCH + kk * TILE,
                    V_LPITCH, gl_CooperativeMatrixLayoutRowMajor);
#else
                coopMatLoad(fragV[kk], hact, vBase + ((KV_BLOCK + kk) * HDT + j) * (TILE * TILE),
                    TILE, gl_CooperativeMatrixLayoutRowMajor);
#endif
            }
            [[unroll]] for (uint qq = 0u; qq < QT; ++qq) {
                [[unroll]] for (uint kk = 0u; kk < KTIL; ++kk) {
                    o[qq][j] = coopMatMulAdd(fragV[kk], fragP[qq][kk], o[qq][j]);
                }
            }
        }
        KV_AFTER_O
#endif
