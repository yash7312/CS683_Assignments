// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
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
                for (int x = 0; x < W; x++) {
                    output_row[x] += input_row[x] * weight;
                }
            }
        }
    }
}
