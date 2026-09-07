#include <immintrin.h>

#include <algorithm>

#include "matmul.h"

namespace {

constexpr int MR = 4;
constexpr int NR = 3;

constexpr long kPanelBytes = 96L * 1024L;
constexpr int kPrefetchLines = 2;
constexpr _mm_hint kPrefetchHint = _MM_HINT_T0;

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline float dot(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    float sum = hsum(acc);
    for (; p < K; ++p) sum += a[p] * b[p];
    return sum;
}

inline void prefetch_rows(const float* first_row, int rows, int ld) {
    for (int r = 0; r < rows; ++r) {
        const char* row = reinterpret_cast<const char*>(first_row + static_cast<long>(r) * ld);
        for (int line = 0; line < kPrefetchLines; ++line)
            _mm_prefetch(row + 64 * line, kPrefetchHint);
    }
}

int panel_rows(int len, int K, int tile) {
    long rows = kPanelBytes / (static_cast<long>(K) * sizeof(float));
    rows -= rows % tile;
    if (rows < tile) rows = tile;
    if (rows > len) rows = len;
    return static_cast<int>(rows);
}

inline void tile_4x3(const float* A, const float* B, float* C,
                     int K, int lda, int ldb, int ldc) {
    const float* a0 = A;
    const float* a1 = A + lda;
    const float* a2 = A + 2 * lda;
    const float* a3 = A + 3 * lda;
    const float* b0 = B;
    const float* b1 = B + ldb;
    const float* b2 = B + 2 * ldb;

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
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

    float s00 = hsum(c00), s01 = hsum(c01), s02 = hsum(c02);
    float s10 = hsum(c10), s11 = hsum(c11), s12 = hsum(c12);
    float s20 = hsum(c20), s21 = hsum(c21), s22 = hsum(c22);
    float s30 = hsum(c30), s31 = hsum(c31), s32 = hsum(c32);

    for (; p < K; ++p) {
        s00 += a0[p] * b0[p]; s01 += a0[p] * b1[p]; s02 += a0[p] * b2[p];
        s10 += a1[p] * b0[p]; s11 += a1[p] * b1[p]; s12 += a1[p] * b2[p];
        s20 += a2[p] * b0[p]; s21 += a2[p] * b1[p]; s22 += a2[p] * b2[p];
        s30 += a3[p] * b0[p]; s31 += a3[p] * b1[p]; s32 += a3[p] * b2[p];
    }

    C[0] = s00;           C[1] = s01;           C[2] = s02;
    C[ldc] = s10;         C[ldc + 1] = s11;     C[ldc + 2] = s12;
    C[2 * ldc] = s20;     C[2 * ldc + 1] = s21; C[2 * ldc + 2] = s22;
    C[3 * ldc] = s30;     C[3 * ldc + 1] = s31; C[3 * ldc + 2] = s32;
}

}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int MC = panel_rows(M, K, MR);
    const int NC = panel_rows(N, K, NR);

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);

        for (int ic = 0; ic < M; ic += MC) {
            const int mc = std::min(MC, M - ic);

            int i = 0;
            for (; i + MR <= mc; i += MR) {
                const float* a = A + static_cast<long>(ic + i) * lda;
                float* c = C + static_cast<long>(ic + i) * ldc + jc;

                if (i + 2 * MR <= mc) prefetch_rows(a + static_cast<long>(MR) * lda, MR, lda);

                int j = 0;
                for (; j + NR <= nc; j += NR) {
                    const float* b = B + static_cast<long>(jc + j) * ldb;
                    if (j + 2 * NR <= nc) prefetch_rows(b + static_cast<long>(NR) * ldb, NR, ldb);
                    tile_4x3(a, b, c + j, K, lda, ldb, ldc);
                }

                for (; j < nc; ++j)
                    for (int r = 0; r < MR; ++r)
                        c[static_cast<long>(r) * ldc + j] =
                            dot(a + static_cast<long>(r) * lda,
                                B + static_cast<long>(jc + j) * ldb, K);
            }

            for (; i < mc; ++i)
                for (int j = 0; j < nc; ++j)
                    C[static_cast<long>(ic + i) * ldc + jc + j] =
                        dot(A + static_cast<long>(ic + i) * lda,
                            B + static_cast<long>(jc + j) * ldb, K);
        }
    }
}
