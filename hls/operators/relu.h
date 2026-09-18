#ifndef MEMBER5_RELU_H
#define MEMBER5_RELU_H

/*
 * ReLU(x) = max(0, x), element-wise.
 * Applied after Conv1, Conv2, FC1, FC2 (never after FC3).
 * Exact in both float and fixed modes (compare + select only).
 */

#include "../../config/types.h"

template <int N>
void relu(const data_t in[N], data_t out[N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        const data_t v = in[i];
        out[i] = (v > data_t(0)) ? v : data_t(0);
    }
}

#endif // MEMBER5_RELU_H
