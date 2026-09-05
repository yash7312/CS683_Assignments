// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    int p = K / 2;
    int stride = W + 2 * p;

    for (int i = 0; i < H * W; i++) {
        out[i] = 0.0f;
    }

    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            float weight = ker[ky * K + kx];
            for (int y = 0; y < H; y++) {
                const float* input_row = in + (y + ky) * stride + kx;
                float* output_row = out + y * W;
                for (int x = 0; x < W; x += 8) {
                    output_row[x] += input_row[x] * weight;
                    output_row[x + 1] += input_row[x + 1] * weight;
                    output_row[x + 2] += input_row[x + 2] * weight;
                    output_row[x + 3] += input_row[x + 3] * weight;
                    output_row[x + 4] += input_row[x + 4] * weight;
                    output_row[x + 5] += input_row[x + 5] * weight;
                    output_row[x + 6] += input_row[x + 6] * weight;
                    output_row[x + 7] += input_row[x + 7] * weight;
                }
            }
        }
    }
}
