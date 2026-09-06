// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // Unroll factor : 4 

    for (int oy = 0; oy < H; ++oy) {
        int ox ; 
        for (ox = 0; ox < W-3; ox+=4) {
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc1 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox + 2 + kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox + 3 + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc1;
            out[oy * W + ox + 1] = acc2;
            out[oy * W + ox + 2] = acc3;
            out[oy * W + ox + 3] = acc4;
        }

        // For the remaining pixel 
        // Ex : W = 2 , then above loop won't execute 
        for (; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
        
    }
}
