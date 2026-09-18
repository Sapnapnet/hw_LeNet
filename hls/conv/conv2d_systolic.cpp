// HLS translation unit for the GEMM-form systolic Conv core.
// Top candidates: conv1_systolic / conv2_systolic.
#include "conv2d_systolic.h"
#include "conv2d_gemm.h"

void conv1_systolic(
    const data_t in[INPUT_SIZE],
    const weight_t w[CONV1_WEIGHT_COUNT],
    data_t out[CONV1_OUT_SIZE])
{
    // 8 output pixels x 6 output channels = 48 PEs.
    conv2d_gemm_direct<8, CONV1_COUT, CONV1_CIN, CONV1_COUT, CONV1_IN_H, CONV1_IN_W,
                CONV1_KH, CONV1_KW>(in, w, out);
}

void conv2_systolic(
    const data_t in[POOL1_OUT_SIZE],
    const weight_t w[CONV2_WEIGHT_COUNT],
    data_t out[CONV2_OUT_SIZE])
{
    // 8 output pixels x 8 output channels = 64 PEs, two channel tiles.
    conv2d_gemm_direct<8, 8, CONV2_CIN, CONV2_COUT, CONV2_IN_H, CONV2_IN_W,
                CONV2_KH, CONV2_KW>(in, w, out);
}
