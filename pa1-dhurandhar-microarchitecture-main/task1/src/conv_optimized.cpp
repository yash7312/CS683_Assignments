// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    int p = K / 2;
    int stride = W + 2 * p;
    const int tile_height = 16;

    for (int start_y = 0; start_y < H; start_y += tile_height) {
        int end_y = start_y + tile_height;
        if (end_y > H) {
            end_y = H;
        }

        for (int y = start_y; y < end_y; y++) {
            float* output_row = out + y * W;
            int x = 0;

            for (; x + 31 < W; x += 32) {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                for (int ky = 0; ky < K; ky++) {
                    const float* input_row = in + (y + ky) * stride + x;
                    for (int kx = 0; kx < K; kx++) {
                        __m256 weight = _mm256_set1_ps(ker[ky * K + kx]);
                        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(input_row + kx), weight, sum0);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(input_row + kx + 8), weight, sum1);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(input_row + kx + 16), weight, sum2);
                        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(input_row + kx + 24), weight, sum3);
                    }
                }

                _mm256_storeu_ps(output_row + x, sum0);
                _mm256_storeu_ps(output_row + x + 8, sum1);
                _mm256_storeu_ps(output_row + x + 16, sum2);
                _mm256_storeu_ps(output_row + x + 24, sum3);
            }

            for (; x < W; x += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (int ky = 0; ky < K; ky++) {
                    const float* input_row = in + (y + ky) * stride + x;
                    for (int kx = 0; kx < K; kx++) {
                        __m256 weight = _mm256_set1_ps(ker[ky * K + kx]);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(input_row + kx), weight, sum);
                    }
                }
                _mm256_storeu_ps(output_row + x, sum);
            }
        }
    }
}
