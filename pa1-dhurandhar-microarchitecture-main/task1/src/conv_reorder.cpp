// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K/2 ; 
    const int in_stride = W + 2*p ; // padded row stride 

    /*
    Case 1 : Outer loop on image, inner loop on kernel : (32 Kb L1 cache, 64 Byte cache line)
    -> 500 cache line can be present
    -> So, for the first row K*([K/16]) lines are fetched and then they are reused, along with the kernel vector.
    -> Assuming K is small, this shouldn't be a problem. 
    Case 2 : Inner loop on image, outer loop on kernel : 
    -> There would be less cache misses are we are iterating linearly, in the above after each roughly 14th interger 
    -> , there would be 3 cache misses, but in this there would be at every W th integer a cache miss.  
    -> Also, assuming that the kernel is small and it fits whithin few cache line, a normal eviciton policy like LRU, won't kick it out 

    // Assuming out is 0 initialised. 
    */

    for(int ky = 0 ; ky < K ; ++ky)
    {
        for(int kx = 0 ; kx < K ; ++kx)
        {
            for(int oy = 0 ; oy < H ; ++oy)
            {
                for(int ox = 0 ; ox < W ; ++ox)
                {
                    out[oy*W + ox] += in[(oy+ky)*in_stride + (ox+kx)] * ker[ky*K + kx] ; 
                }
            }
        }
    }
}
