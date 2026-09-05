// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    int p = K / 2;
    int stride = W + 2 * p;
    const int tile_height = 2;
    __m256 zero = _mm256_setzero_ps();

    for (int start_y = 0; start_y < H; start_y += tile_height) {
        int end_y = start_y + tile_height;
        if (end_y > H) {
            end_y = H;
        }

        for (int y = start_y; y < end_y; y++) {
            float* output_row = out + y * W;
            for (int x = 0; x < W; x += 8) {
                _mm256_storeu_ps(output_row + x, zero);
            }
        }

        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {
                __m256 weight = _mm256_set1_ps(ker[ky * K + kx]);
                for (int y = start_y; y < end_y; y++) {
                    const float* input_row = in + (y + ky) * stride + kx;
                    float* output_row = out + y * W;
                    for (int x = 0; x < W; x += 8) {
                        __m256 input_values = _mm256_loadu_ps(input_row + x);
                        __m256 output_values = _mm256_loadu_ps(output_row + x);
                        output_values = _mm256_fmadd_ps(input_values, weight, output_values);
                        _mm256_storeu_ps(output_row + x, output_values);
                    }
                }
            }
        }
    }
}
