#ifndef MEMBER5_FLATTEN_H
#define MEMBER5_FLATTEN_H

/*
 * Flatten [C, H, W] -> [C*H*W].
 * Storage is already C-contiguous CHW (member1 frozen order:
 * index = c*H*W + h*W + w), so this layer is an explicit identity copy that
 * marks the feature-map -> vector boundary.
 */

#include "../../config/types.h"

template <int C, int H, int W>
void flatten(const data_t in[C * H * W], data_t out[C * H * W]) {
#pragma HLS INLINE
    for (int i = 0; i < C * H * W; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = in[i];
    }
}

#endif // MEMBER5_FLATTEN_H
