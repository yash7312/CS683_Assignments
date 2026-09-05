// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING
//
// Same 4x2 register-tiled AVX2/FMA micro-kernel as Stage 1, but M and N are now blocked
// into panels sized to stay resident in L2: for a tile, the A-panel (MC rows x K) and the
// B-panel (NC rows x K) are re-used across the whole tile instead of streaming the full
// N (or M) sweep past a fixed row-block, which is what made Stage 1 memory-bound once the
// working set exceeded cache. Software prefetch hints are layered on top: within the K
// reduction (to hide the latency of the next cache line of each streamed row), and across
// register-tile / row-block boundaries (to hide the latency of jumping to a new row's
// non-adjacent address, which the hardware's next-line prefetcher does not track).

#include <immintrin.h>

#include <algorithm>

#include "matmul.h"

namespace {

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float dot(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8) {
        __m256 va = _mm256_loadu_ps(a + p);
        __m256 vb = _mm256_loadu_ps(b + p);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum256(acc);
    for (; p < K; ++p) sum += a[p] * b[p];
    return sum;
}

constexpr int MR = 4;
constexpr int NR = 2;

// How far ahead (in floats) to prefetch within the current K-reduction.
constexpr int kPFDistFloats = 64;

// Panel byte budget per operand: chosen so an A-panel and a B-panel (each this many bytes)
// co-reside comfortably inside a ~1.25 MiB per-core L2 (measured via lscpu/getconf on the
// dev machine), leaving headroom for the C tile and associativity conflicts.
constexpr long kPanelByteBudget = 192L * 1024L;

int pick_panel(int dimLen, int K, int reg_tile) {
    long rows = kPanelByteBudget / (static_cast<long>(K) * sizeof(float));
    int panel = static_cast<int>(rows);
    if (panel < reg_tile) panel = reg_tile;
    if (panel > dimLen) panel = dimLen;
    panel -= panel % reg_tile;
    if (panel == 0) panel = std::min(reg_tile, dimLen);
    return panel;
}

}  // namespace

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int NC = pick_panel(N, K, NR);
    const int MC = pick_panel(M, K, MR);

    for (int jc = 0; jc < N; jc += NC) {
        const int jb = std::min(NC, N - jc);
        for (int ic = 0; ic < M; ic += MC) {
            const int ib = std::min(MC, M - ic);

            int i = 0;
            for (; i + MR <= ib; i += MR) {
                const int gi = ic + i;
                const float* a0 = A + static_cast<long>(gi + 0) * lda;
                const float* a1 = A + static_cast<long>(gi + 1) * lda;
                const float* a2 = A + static_cast<long>(gi + 2) * lda;
                const float* a3 = A + static_cast<long>(gi + 3) * lda;

                int j = 0;
                for (; j + NR <= jb; j += NR) {
                    const int gj = jc + j;
                    const float* b0 = B + static_cast<long>(gj + 0) * ldb;
                    const float* b1 = B + static_cast<long>(gj + 1) * ldb;

                    // Prefetch the NEXT register tile's row(s): a jump to a new A/B row is
                    // a new cache line/page, unlike the sequential K-stream within a row
                    // that the hardware prefetcher already tracks.
                    if (j + NR < jb) {
                        _mm_prefetch(reinterpret_cast<const char*>(
                                         B + static_cast<long>(jc + j + NR) * ldb),
                                     _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(
                                         B + static_cast<long>(jc + j + NR + 1) * ldb),
                                     _MM_HINT_T0);
                    } else if (i + MR < ib) {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(A + static_cast<long>(gi + MR) * lda),
                            _MM_HINT_T1);
                    }

                    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

                    int p = 0;
                    for (; p + 8 <= K; p += 8) {
                        _mm_prefetch(reinterpret_cast<const char*>(a0 + p + kPFDistFloats),
                                     _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(a1 + p + kPFDistFloats),
                                     _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(a2 + p + kPFDistFloats),
                                     _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(a3 + p + kPFDistFloats),
                                     _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(b0 + p + kPFDistFloats),
                                     _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(b1 + p + kPFDistFloats),
                                     _MM_HINT_T0);

                        __m256 va0 = _mm256_loadu_ps(a0 + p);
                        __m256 va1 = _mm256_loadu_ps(a1 + p);
                        __m256 va2 = _mm256_loadu_ps(a2 + p);
                        __m256 va3 = _mm256_loadu_ps(a3 + p);
                        __m256 vb0 = _mm256_loadu_ps(b0 + p);
                        __m256 vb1 = _mm256_loadu_ps(b1 + p);

                        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                        acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                        acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                        acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                        acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                    }

                    float c00 = hsum256(acc00), c01 = hsum256(acc01);
                    float c10 = hsum256(acc10), c11 = hsum256(acc11);
                    float c20 = hsum256(acc20), c21 = hsum256(acc21);
                    float c30 = hsum256(acc30), c31 = hsum256(acc31);

                    for (; p < K; ++p) {
                        c00 += a0[p] * b0[p]; c01 += a0[p] * b1[p];
                        c10 += a1[p] * b0[p]; c11 += a1[p] * b1[p];
                        c20 += a2[p] * b0[p]; c21 += a2[p] * b1[p];
                        c30 += a3[p] * b0[p]; c31 += a3[p] * b1[p];
                    }

                    C[static_cast<long>(gi + 0) * ldc + gj + 0] = c00;
                    C[static_cast<long>(gi + 0) * ldc + gj + 1] = c01;
                    C[static_cast<long>(gi + 1) * ldc + gj + 0] = c10;
                    C[static_cast<long>(gi + 1) * ldc + gj + 1] = c11;
                    C[static_cast<long>(gi + 2) * ldc + gj + 0] = c20;
                    C[static_cast<long>(gi + 2) * ldc + gj + 1] = c21;
                    C[static_cast<long>(gi + 3) * ldc + gj + 0] = c30;
                    C[static_cast<long>(gi + 3) * ldc + gj + 1] = c31;
                }
                // Leftover column within this tile.
                for (; j < jb; ++j) {
                    const int gj = jc + j;
                    const float* b = B + static_cast<long>(gj) * ldb;
                    C[static_cast<long>(gi + 0) * ldc + gj] = dot(a0, b, K);
                    C[static_cast<long>(gi + 1) * ldc + gj] = dot(a1, b, K);
                    C[static_cast<long>(gi + 2) * ldc + gj] = dot(a2, b, K);
                    C[static_cast<long>(gi + 3) * ldc + gj] = dot(a3, b, K);
                }
            }
            // Leftover rows within this tile.
            for (; i < ib; ++i) {
                const int gi = ic + i;
                const float* a = A + static_cast<long>(gi) * lda;
                for (int j = 0; j < jb; ++j) {
                    const int gj = jc + j;
                    const float* b = B + static_cast<long>(gj) * ldb;
                    C[static_cast<long>(gi) * ldc + gj] = dot(a, b, K);
                }
            }
        }
    }
}
