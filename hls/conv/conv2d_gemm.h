#ifndef CONV2D_GEMM_H
#define CONV2D_GEMM_H

/*
 * Convolution as direct im2col-fused output-stationary GEMM.
 *
 * Output-stationary PE array: M spatial outputs x N_TILE output channels.
 */

#include "../../config/network_config.h"
#include "gemm_systolic.h"

// ---------------------------------------------------------------------------
// Direct fused conv-GEMM: no explicit im2col A buffer.
//
// The im2col read is fused into the (ic,kh,kw) loop. For each output-pixel
// tile, the M output positions' row/col are precomputed, then the PE array
// directly consumes input values from the feature map. This removes the
// A-tile build overhead which dominates small-K convs such as Conv1.
// ---------------------------------------------------------------------------
template <int M, int N_TILE, int CIN, int COUT, int IN_H, int IN_W, int KH, int KW>
void conv2d_gemm_direct(
    const data_t in[CIN * IN_H * IN_W],
    const weight_t w[COUT * CIN * KH * KW],
    data_t out[COUT * (IN_H - KH + 1) * (IN_W - KW + 1)])
{
    const int K  = CIN * KH * KW;
    const int OH = IN_H - KH + 1;
    const int OW = IN_W - KW + 1;
    const int SP = OH * OW;

    unsigned char oh[M];   // narrow indices (OH/OW <= 24) -> tiny addressing logic
    unsigned char ow[M];
    systolic_acc_t acc[M][N_TILE];
#pragma HLS ARRAY_PARTITION variable=oh complete dim=1
#pragma HLS ARRAY_PARTITION variable=ow complete dim=1
#pragma HLS ARRAY_PARTITION variable=w block factor=N_TILE dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (int base = 0; base < SP; base += M) {
        for (int m = 0; m < M; ++m) {
            oh[m] = (base + m) / OW;
            ow[m] = (base + m) % OW;
        }

        for (int oc0 = 0; oc0 < COUT; oc0 += N_TILE) {
            for (int m = 0; m < M; ++m) {
                for (int j = 0; j < N_TILE; ++j) {
                    acc[m][j] = systolic_acc_t(0);
                }
            }

            for (int ic = 0; ic < CIN; ++ic) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
#pragma HLS PIPELINE II=1
                        const int k = (ic * KH + kh) * KW + kw;
                        data_t  av_loc[M];
                        weight_t wv_loc[N_TILE];
#pragma HLS ARRAY_PARTITION variable=av_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=wv_loc complete dim=1
                        for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL
                            av_loc[m] =
                                in[(ic * IN_H + oh[m] + kh) * IN_W + ow[m] + kw];
                        }
                        for (int j = 0; j < N_TILE; ++j) {
#pragma HLS UNROLL
                            wv_loc[j] = w[(oc0 + j) * K + k];
                        }
                        for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL
                            for (int j = 0; j < N_TILE; ++j) {
#pragma HLS UNROLL
#if defined(LENET_USE_FIXED) && defined(LENET_ACC_INT)
                                ap_int<LENET_DATA_W>   a_raw = av_loc[m].range(LENET_DATA_W - 1, 0);
                                ap_int<LENET_WEIGHT_W> b_raw = wv_loc[j].range(LENET_WEIGHT_W - 1, 0);
                                acc[m][j] += (ap_int<LENET_DATA_W + LENET_WEIGHT_W>)(a_raw * b_raw);
#else
                                acc[m][j] += av_loc[m] * wv_loc[j];
#endif
                            }
                        }
                    }
                }
            }

            for (int m = 0; m < M; ++m) {
                for (int j = 0; j < N_TILE; ++j) {
#if defined(LENET_USE_FIXED) && defined(LENET_ACC_INT)
                    out[(oc0 + j) * SP + base + m] = quantize_acc(acc[m][j]);
#else
                    out[(oc0 + j) * SP + base + m] = data_t(acc[m][j]);
#endif
                }
            }
        }
    }
}

#endif // CONV2D_GEMM_H
