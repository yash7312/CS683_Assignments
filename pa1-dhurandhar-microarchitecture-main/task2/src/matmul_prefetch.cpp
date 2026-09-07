// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

static const int TILE_ROWS = 4;
static const int TILE_COLS = 3;
static const long PANEL_BYTES = 96L * 1024L;
static const int PREFETCH_LINES = 2;
static const int CACHE_LINE_BYTES = 64;

static float horizontal_sum(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static float dot_product(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    }
    float sum = horizontal_sum(acc);
    for (; p < K; p++) {
        sum += a[p] * b[p];
    }
    return sum;
}

static void prefetch_rows(const float* first_row, int rows, int ld) {
    for (int r = 0; r < rows; r++) {
        const char* row = reinterpret_cast<const char*>(first_row + static_cast<long>(r) * ld);
        for (int line = 0; line < PREFETCH_LINES; line++) {
            _mm_prefetch(row + line * CACHE_LINE_BYTES, _MM_HINT_T0);
        }
    }
}

static int panel_rows(int total_rows, int K, int tile_rows) {
    long rows = PANEL_BYTES / (static_cast<long>(K) * sizeof(float));
    rows -= rows % tile_rows;
    if (rows < tile_rows) rows = tile_rows;
    if (rows > total_rows) rows = total_rows;
    return static_cast<int>(rows);
}

static void tile_4x3(const float* a, const float* b, float* c,
                     int K, int lda, int ldb, int ldc) {
    const float* a0 = a;
    const float* a1 = a + lda;
    const float* a2 = a + 2 * lda;
    const float* a3 = a + 3 * lda;
    const float* b0 = b;
    const float* b1 = b + ldb;
    const float* b2 = b + 2 * ldb;

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
        __m256 vb0 = _mm256_loadu_ps(b0 + p);
        __m256 vb1 = _mm256_loadu_ps(b1 + p);
        __m256 vb2 = _mm256_loadu_ps(b2 + p);

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

    float s00 = horizontal_sum(c00), s01 = horizontal_sum(c01), s02 = horizontal_sum(c02);
    float s10 = horizontal_sum(c10), s11 = horizontal_sum(c11), s12 = horizontal_sum(c12);
    float s20 = horizontal_sum(c20), s21 = horizontal_sum(c21), s22 = horizontal_sum(c22);
    float s30 = horizontal_sum(c30), s31 = horizontal_sum(c31), s32 = horizontal_sum(c32);

    for (; p < K; p++) {
        s00 += a0[p] * b0[p]; s01 += a0[p] * b1[p]; s02 += a0[p] * b2[p];
        s10 += a1[p] * b0[p]; s11 += a1[p] * b1[p]; s12 += a1[p] * b2[p];
        s20 += a2[p] * b0[p]; s21 += a2[p] * b1[p]; s22 += a2[p] * b2[p];
        s30 += a3[p] * b0[p]; s31 += a3[p] * b1[p]; s32 += a3[p] * b2[p];
    }

    c[0] = s00;           c[1] = s01;           c[2] = s02;
    c[ldc] = s10;         c[ldc + 1] = s11;     c[ldc + 2] = s12;
    c[2 * ldc] = s20;     c[2 * ldc + 1] = s21; c[2 * ldc + 2] = s22;
    c[3 * ldc] = s30;     c[3 * ldc + 1] = s31; c[3 * ldc + 2] = s32;
}

static void multiply_panel(const float* A, const float* B, float* C,
                           int rows, int cols, int K, int lda, int ldb, int ldc) {
    int i = 0;
    for (; i + TILE_ROWS <= rows; i += TILE_ROWS) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;

        if (i + 2 * TILE_ROWS <= rows) {
            prefetch_rows(a + static_cast<long>(TILE_ROWS) * lda, TILE_ROWS, lda);
        }

        int j = 0;
        for (; j + TILE_COLS <= cols; j += TILE_COLS) {
            const float* b = B + static_cast<long>(j) * ldb;
            if (j + 2 * TILE_COLS <= cols) {
                prefetch_rows(b + static_cast<long>(TILE_COLS) * ldb, TILE_COLS, ldb);
            }
            tile_4x3(a, b, c + j, K, lda, ldb, ldc);
        }
        for (; j < cols; j++) {
            const float* b = B + static_cast<long>(j) * ldb;
            for (int r = 0; r < TILE_ROWS; r++) {
                c[r * ldc + j] = dot_product(a + r * lda, b, K);
            }
        }
    }

    for (; i < rows; i++) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < cols; j++) {
            const float* b = B + static_cast<long>(j) * ldb;
            c[j] = dot_product(a, b, K);
        }
    }
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int MC = panel_rows(M, K, TILE_ROWS);
    const int NC = panel_rows(N, K, TILE_COLS);

    for (int jc = 0; jc < N; jc += NC) {
        int nc = NC;
        if (jc + nc > N) nc = N - jc;

        for (int ic = 0; ic < M; ic += MC) {
            int mc = MC;
            if (ic + mc > M) mc = M - ic;

            const float* a = A + static_cast<long>(ic) * lda;
            const float* b = B + static_cast<long>(jc) * ldb;
            float* c = C + static_cast<long>(ic) * ldc + jc;
            multiply_panel(a, b, c, mc, nc, K, lda, ldb, ldc);
        }
    }
}
