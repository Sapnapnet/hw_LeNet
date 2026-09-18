#ifndef CONV2D_SYSTOLIC_H
#define CONV2D_SYSTOLIC_H

/*
 * Systolic Conv/MAC core - output-stationary GEMM form.
 *
 * The actual implementation lives in:
 *   gemm_systolic.h  : reusable output-stationary GEMM PE array
 *   conv2d_gemm.h    : Conv as direct im2col-fused GEMM
 *
 * These two wrappers are the concrete HLS top entry points for Conv1/Conv2.
 * Member 5 can reuse gemm_systolic.h directly for FC (y = W*x).
 */

#include "conv2d_gemm.h"

void conv1_systolic(
    const data_t in[INPUT_SIZE],
    const weight_t w[CONV1_WEIGHT_COUNT],
    data_t out[CONV1_OUT_SIZE]);

void conv2_systolic(
    const data_t in[POOL1_OUT_SIZE],
    const weight_t w[CONV2_WEIGHT_COUNT],
    data_t out[CONV2_OUT_SIZE]);

#endif // CONV2D_SYSTOLIC_H
