// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

static const int TILE_ROWS = 4;
static const int TILE_COLS = 3;
static const long PANEL_BYTES = 192L * 1024L;
static const int MAX_K_BLOCK = 1024;
static const int PREFETCH_LINES = 1;
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

static void store(float* c, float value, bool accumulate) {
    if (accumulate) {
        *c += value;
    } else {
        *c = value;
    }
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
                     int K, int lda, int ldb, int ldc, bool accumulate) {
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

    float* c0 = c;
    float* c1 = c + ldc;
    float* c2 = c + 2 * ldc;
    float* c3 = c + 3 * ldc;
    store(c0, s00, accumulate); store(c0 + 1, s01, accumulate); store(c0 + 2, s02, accumulate);
    store(c1, s10, accumulate); store(c1 + 1, s11, accumulate); store(c1 + 2, s12, accumulate);
    store(c2, s20, accumulate); store(c2 + 1, s21, accumulate); store(c2 + 2, s22, accumulate);
    store(c3, s30, accumulate); store(c3 + 1, s31, accumulate); store(c3 + 2, s32, accumulate);
}

static void multiply_panel(const float* A, const float* B, float* C,
                           int rows, int cols, int K, int lda, int ldb, int ldc,
                           bool accumulate) {
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
            tile_4x3(a, b, c + j, K, lda, ldb, ldc, accumulate);
        }
        for (; j < cols; j++) {
            const float* b = B + static_cast<long>(j) * ldb;
            for (int r = 0; r < TILE_ROWS; r++) {
                store(c + r * ldc + j, dot_product(a + r * lda, b, K), accumulate);
            }
        }
    }

    for (; i < rows; i++) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < cols; j++) {
            const float* b = B + static_cast<long>(j) * ldb;
            store(c + j, dot_product(a, b, K), accumulate);
        }
    }
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    int KC = K;
    if (KC > MAX_K_BLOCK) KC = MAX_K_BLOCK;
    const int NC = panel_rows(N, KC, TILE_COLS);

    for (int pc = 0; pc < K; pc += KC) {
        int kc = KC;
        if (pc + kc > K) kc = K - pc;
        const bool accumulate = pc > 0;

        for (int jc = 0; jc < N; jc += NC) {
            int nc = NC;
            if (jc + nc > N) nc = N - jc;

            const float* a = A + pc;
            const float* b = B + static_cast<long>(jc) * ldb + pc;
            float* c = C + jc;
            multiply_panel(a, b, c, M, nc, kc, lda, ldb, ldc, accumulate);
        }
    }
}
