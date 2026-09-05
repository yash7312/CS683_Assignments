// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    int p = K / 2;
    int stride = W + 2 * p;
    const int tile_height = 1;

    for (int start_y = 0; start_y < H; start_y += tile_height) {
        int end_y = start_y + tile_height;
        if (end_y > H) {
            end_y = H;
        }

        for (int y = start_y; y < end_y; y++) {
            float* output_row = out + y * W;
            for (int x = 0; x < W; x += 8) {
                output_row[x] = 0.0f;
                output_row[x + 1] = 0.0f;
                output_row[x + 2] = 0.0f;
                output_row[x + 3] = 0.0f;
                output_row[x + 4] = 0.0f;
                output_row[x + 5] = 0.0f;
                output_row[x + 6] = 0.0f;
                output_row[x + 7] = 0.0f;
            }
        }

        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {
                float weight = ker[ky * K + kx];
                for (int y = start_y; y < end_y; y++) {
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
}
