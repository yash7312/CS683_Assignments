// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// Every parameter below was chosen by measurement on the target machine (i7-13700HX:
// 48 KiB L1d, 1.25 MiB L2 per core, 30 MiB shared L3), not by guesswork.
//
// 1. SIMD + register blocking (4x3).  A 4x3 tile of C lives in 12 AVX2 accumulators; each
//    K-step loads 3 B vectors and 4 A vectors and issues 12 FMAs, so the FMA:load ratio is
//    12:7 = 1.71 -- the best achievable within the 16-register AVX2 file. Each A vector is
//    dead after its 3 FMAs, so only one is live at a time: 12 accumulators + 3 B + 1 A =
//    exactly 16 ymm registers. Measured: 4x3 = 128 GFLOP/s, 6x2 = 119, 4x2 = 110, while
//    5x3 and 4x4 need 19-21 registers and collapse to 94-101 GFLOP/s from spilling.
//
// 2. Unrolling.  The tile is written out by hand. An equivalent version using
//    `__m256 acc[MR][NR]` with counted loops measured ~50 GFLOP/s, because -O2 (no
//    -funroll-loops, which the pinned flags do not allow) leaves the accumulator array in
//    memory instead of registers.
//
// 3. Loop reordering + cache tiling.  N is cut into panels of NC rows of B and the ENTIRE
//    M dimension is swept inside one panel, so the B panel (NC x KC floats) stays resident
//    in L2 and is reused by every row of A. Memory traffic drops from ~4*M*N*K/MR bytes to
//    ~4*M*N*K/NC -- about 1 GB down to ~34 MB at 1024^3. Measured at 1024^3: NC=126 gives
//    132 GFLOP/s vs 100 GFLOP/s with no N-blocking.
//
// 4. Software prefetch.  Deliberately SPARSE. Within a row the K-loop is a unit-stride
//    stream the hardware prefetcher already tracks; what it cannot follow is the jump to a
//    new row. So the only hint is at the A row-block boundary, where it is amortized over
//    ~NC/NR inner iterations and costs nothing measurable. Stage 2 shows the alternative:
//    prefetching every K-iteration adds 6 uops against 6 loads + 8 FMAs, turning an
//    FMA-bound loop into a load-port-bound one, and it measured SLOWER than no prefetch.
//
// 5. K-tiling.  Only for very large K, to keep the MR A-rows inside L1. It is off at
//    K <= kMaxKC because splitting K forces C to be read-modified-written once per K-block:
//    at 1024^3, KC=256 measured 103 GFLOP/s against 131 untiled.

#include <immintrin.h>

#include <algorithm>

#include "matmul.h"

namespace {

// Horizontal sum of 8 lanes (AVX + SSE3 reduction sequence).
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

// Vectorized dot product over a K-range with scalar tail; used by the cleanup paths.
inline float dot_range(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= n; p += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    float s = hsum256(acc);
    for (; p < n; ++p) s += a[p] * b[p];
    return s;
}

// The first K-block stores; later K-blocks accumulate.
inline void put(float* p, float v, bool first) {
    if (first) *p = v; else *p += v;
}

constexpr int MR = 4;  // rows of C per register tile
constexpr int NR = 3;  // cols of C per register tile

// Byte budget for the L2-resident B panel: a conservative slice of the 1.25 MiB per-core
// L2, leaving room for the streamed A rows and the C tile. Yields NC = 126 at K = 1024,
// which measured at the top of a flat 96..150 optimum.
constexpr long kPanelBytes = 512L * 1024L;

// K-tile cap. Above this, MR A-rows would no longer fit in the 48 KiB L1.
constexpr int kMaxKC = 2048;

}  // namespace

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int KC = std::min(K, kMaxKC);

    int NC = static_cast<int>(kPanelBytes / (static_cast<long>(KC) * sizeof(float)));
    if (NC > N) NC = N;
    NC -= NC % NR;
    if (NC < NR) NC = NR;

    for (int pc = 0; pc < K; pc += KC) {
        const int kb = std::min(KC, K - pc);
        const bool first = (pc == 0);

        for (int jc = 0; jc < N; jc += NC) {
            const int jn = std::min(NC, N - jc);

            int i = 0;
            for (; i + MR <= M; i += MR) {
                const float* a0 = A + static_cast<long>(i + 0) * lda + pc;
                const float* a1 = A + static_cast<long>(i + 1) * lda + pc;
                const float* a2 = A + static_cast<long>(i + 2) * lda + pc;
                const float* a3 = A + static_cast<long>(i + 3) * lda + pc;

                // The one prefetch site: the head of the next A row-block. A new row is a
                // far jump the hardware next-line prefetcher will not follow, and this hint
                // is amortized over the whole j-loop below.
                if (i + 2 * MR <= M) {
                    const float* an = A + static_cast<long>(i + MR) * lda + pc;
                    _mm_prefetch(reinterpret_cast<const char*>(an), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(an + lda), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(an + 2 * lda), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(an + 3 * lda), _MM_HINT_T0);
                }

                int j = 0;
                for (; j + NR <= jn; j += NR) {
                    const float* b0 = B + static_cast<long>(jc + j + 0) * ldb + pc;
                    const float* b1 = B + static_cast<long>(jc + j + 1) * ldb + pc;
                    const float* b2 = B + static_cast<long>(jc + j + 2) * ldb + pc;

                    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(),
                           c02 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(),
                           c12 = _mm256_setzero_ps();
                    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(),
                           c22 = _mm256_setzero_ps();
                    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(),
                           c32 = _mm256_setzero_ps();

                    int p = 0;
                    for (; p + 8 <= kb; p += 8) {
                        const __m256 vb0 = _mm256_loadu_ps(b0 + p);
                        const __m256 vb1 = _mm256_loadu_ps(b1 + p);
                        const __m256 vb2 = _mm256_loadu_ps(b2 + p);

                        __m256 va = _mm256_loadu_ps(a0 + p);
                        c00 = _mm256_fmadd_ps(va, vb0, c00);
                        c01 = _mm256_fmadd_ps(va, vb1, c01);
                        c02 = _mm256_fmadd_ps(va, vb2, c02);

                        va = _mm256_loadu_ps(a1 + p);
                        c10 = _mm256_fmadd_ps(va, vb0, c10);
                        c11 = _mm256_fmadd_ps(va, vb1, c11);
                        c12 = _mm256_fmadd_ps(va, vb2, c12);

                        va = _mm256_loadu_ps(a2 + p);
                        c20 = _mm256_fmadd_ps(va, vb0, c20);
                        c21 = _mm256_fmadd_ps(va, vb1, c21);
                        c22 = _mm256_fmadd_ps(va, vb2, c22);

                        va = _mm256_loadu_ps(a3 + p);
                        c30 = _mm256_fmadd_ps(va, vb0, c30);
                        c31 = _mm256_fmadd_ps(va, vb1, c31);
                        c32 = _mm256_fmadd_ps(va, vb2, c32);
                    }

                    float s[MR][NR] = {{hsum256(c00), hsum256(c01), hsum256(c02)},
                                       {hsum256(c10), hsum256(c11), hsum256(c12)},
                                       {hsum256(c20), hsum256(c21), hsum256(c22)},
                                       {hsum256(c30), hsum256(c31), hsum256(c32)}};

                    // Scalar tail for K % 8.
                    const float* ar[MR] = {a0, a1, a2, a3};
                    const float* br[NR] = {b0, b1, b2};
                    for (; p < kb; ++p)
                        for (int r = 0; r < MR; ++r)
                            for (int c = 0; c < NR; ++c) s[r][c] += ar[r][p] * br[c][p];

                    for (int r = 0; r < MR; ++r)
                        for (int c = 0; c < NR; ++c)
                            put(C + static_cast<long>(i + r) * ldc + jc + j + c, s[r][c],
                                first);
                }
                // Leftover columns of this panel (jn % NR), for the current MR rows.
                for (; j < jn; ++j) {
                    const float* b = B + static_cast<long>(jc + j) * ldb + pc;
                    put(C + static_cast<long>(i + 0) * ldc + jc + j, dot_range(a0, b, kb), first);
                    put(C + static_cast<long>(i + 1) * ldc + jc + j, dot_range(a1, b, kb), first);
                    put(C + static_cast<long>(i + 2) * ldc + jc + j, dot_range(a2, b, kb), first);
                    put(C + static_cast<long>(i + 3) * ldc + jc + j, dot_range(a3, b, kb), first);
                }
            }
            // Leftover rows (M % MR).
            for (; i < M; ++i) {
                const float* a = A + static_cast<long>(i) * lda + pc;
                for (int j = 0; j < jn; ++j) {
                    const float* b = B + static_cast<long>(jc + j) * ldb + pc;
                    put(C + static_cast<long>(i) * ldc + jc + j, dot_range(a, b, kb), first);
                }
            }
        }
    }
}
