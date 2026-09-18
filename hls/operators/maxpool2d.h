#ifndef MEMBER5_MAXPOOL2D_H
#define MEMBER5_MAXPOOL2D_H

/*
 * MaxPool 2x2, stride 2, padding 0 (non-overlapping), CHW layout.
 * [C, H, W] -> [C, H/2, W/2]; H and W must be even.
 * Exact in both float and fixed modes (comparisons only).
 */

#include "../../config/types.h"

template <int C, int H, int W>
void maxpool2d(const data_t in[C * H * W], data_t out[C * (H / 2) * (W / 2)]) {
#pragma HLS INLINE
    static_assert(H % 2 == 0 && W % 2 == 0, "maxpool2d requires even H and W");
    const int OH = H / 2;
    const int OW = W / 2;
    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
#pragma HLS PIPELINE II=1
                const int r = c * H * W + 2 * oh * W + 2 * ow;
                const data_t m01 = (in[r] > in[r + 1]) ? in[r] : in[r + 1];
                const data_t m23 = (in[r + W] > in[r + W + 1]) ? in[r + W] : in[r + W + 1];
                out[c * OH * OW + oh * OW + ow] = (m01 > m23) ? m01 : m23;
            }
        }
    }
}

#endif // MEMBER5_MAXPOOL2D_H
