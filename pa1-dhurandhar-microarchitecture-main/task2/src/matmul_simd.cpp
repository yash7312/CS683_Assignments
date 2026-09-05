// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
//
// Register-tiled (4 A-rows x 2 B-rows) dot-product micro-kernel. For each K-chunk of 8
// floats, 4 A vectors and 2 B vectors are loaded once and reused across all 4x2=8 FMA
// accumulators, so 6 loads service 8 FMAs per step instead of 8 independent dot products
// each reloading their own operands. No cache blocking here (that is Stage 2) -- just
// register blocking, 8-wide AVX2/FMA vectorization of the K reduction, and scalar cleanup
// for M % 4, N % 2, and K % 8 remainders and arbitrary strides.

#include <immintrin.h>

#include "matmul.h"

namespace {

// Horizontal sum of 8 lanes (Peter Cordes' standard AVX+SSE3 reduction sequence).
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

// Single AVX2/FMA dot product with scalar tail, used only for leftover rows/cols.
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

}  // namespace

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int MR = 4;
    constexpr int NR = 2;

    int i = 0;
    for (; i + MR <= M; i += MR) {
        const float* a0 = A + static_cast<long>(i + 0) * lda;
        const float* a1 = A + static_cast<long>(i + 1) * lda;
        const float* a2 = A + static_cast<long>(i + 2) * lda;
        const float* a3 = A + static_cast<long>(i + 3) * lda;

        int j = 0;
        for (; j + NR <= N; j += NR) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;

            __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

            int p = 0;
            for (; p + 8 <= K; p += 8) {
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

            C[static_cast<long>(i + 0) * ldc + j + 0] = c00;
            C[static_cast<long>(i + 0) * ldc + j + 1] = c01;
            C[static_cast<long>(i + 1) * ldc + j + 0] = c10;
            C[static_cast<long>(i + 1) * ldc + j + 1] = c11;
            C[static_cast<long>(i + 2) * ldc + j + 0] = c20;
            C[static_cast<long>(i + 2) * ldc + j + 1] = c21;
            C[static_cast<long>(i + 3) * ldc + j + 0] = c30;
            C[static_cast<long>(i + 3) * ldc + j + 1] = c31;
        }
        // Leftover column (N % NR) for this row-block.
        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            C[static_cast<long>(i + 0) * ldc + j] = dot(a0, b, K);
            C[static_cast<long>(i + 1) * ldc + j] = dot(a1, b, K);
            C[static_cast<long>(i + 2) * ldc + j] = dot(a2, b, K);
            C[static_cast<long>(i + 3) * ldc + j] = dot(a3, b, K);
        }
    }
    // Leftover rows (M % MR).
    for (; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            C[static_cast<long>(i) * ldc + j] = dot(a, b, K);
        }
    }
}
