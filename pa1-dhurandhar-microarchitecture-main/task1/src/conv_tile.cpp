// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);

    /*
    Why tiling benefits ? Assuming that K is not very large 
    -> Consider that when we are computing for the 0th row , we fetch K/2 rows
    - from below, and now when computing for the 1st row K/2 - 1 of these rows are required again 
    - No it would be great if we made the every use of these fetched row at once by reordering the computation somehow 
    - Instead of fetching it again and again, that the idea of tiling. 
    */

    /*
    Block Sie : B 
    Then B*(B/16) cache lines in total if they fit in the DRAM then we are good 
    Given a 32 Kb Cache , B of 64 would suffice : 256 cache lines, and our cache can 
    accomodate 500 cache lines 
    For a complete covulation of B*B block we need B+2p * B+2p data 
    
    we Can also do the partial convultion for the starting elements 
    
    */


}
