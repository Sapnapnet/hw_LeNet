#ifndef GEMM_SYSTOLIC_H
#define GEMM_SYSTOLIC_H

/*
 * Output-stationary GEMM systolic core.
 *
 *   C[M][N] = A[M][K] * B[N][K]^T
 *
 * Layouts (all row-major, compatible with frozen network):
 *   A : [M * K]   input patches / activations
 *   B : [N * K]   weights: [OUT][IN_CH] (conv OIHW or FC [OUT][IN])
 *   C : [M * N]   output channels x spatial / batch dimension
 *
 * Each PE (i,j) holds/accumulates C[i][j] (result-stationary). The k loop is a
 * pipelined systolic GEMM pass; i/j loops are unrolled into a PE array.
 *
 * LUT note: under LENET_ACC_INT the accumulator is a plain integer. The full 12s x 12s -> 24-bit product is added with zero rounding,
 * and the single ">> 11 + round-half-to-even + saturate" pass happens only at
 * the write-back below. This removes the per-MAC ap_fixed rounding/saturation
 * chains that dominated LUT usage in the legacy ap_fixed accumulator.
 */

#include "../../config/types.h"

/*
 * Member-4 local accumulator type for the LUT-optimized integer-accumulator
 * path. This intentionally does NOT modify the shared types.h.
 */
#if defined(LENET_USE_FIXED) && defined(LENET_ACC_INT)
#include <ap_int.h>
typedef ap_int<33> systolic_acc_t;
#else
typedef acc_t systolic_acc_t;
#endif

/*
 * Fixed-point alignment: data_t has F = LENET_DATA_W - LENET_DATA_I (=5),
 * weight_t has F = LENET_WEIGHT_W - LENET_WEIGHT_I (=11), so every product
 * carries F = 5 + 11 = 16 fractional bits. The integer accumulator is
 * therefore a Q16 raw sum. To write back into data_t (F=5) we must drop the
 * low (16 - 5) = 11 bits with round-half-to-even, then saturate to the
 * 12-bit signed range. This is exactly the frozen member-3 rule verified in
 * tests/tb_fc.cpp narrow().
 *
 * These helpers only exist in the fixed-point integer-accumulator build
 * (LENET_USE_FIXED && LENET_ACC_INT); in float mode acc_t/data_t are plain
 * float and the original float MAC path below is used unchanged.
 */
#if defined(LENET_USE_FIXED) && defined(LENET_ACC_INT)
static const int ACC_FRAC = (LENET_DATA_W - LENET_DATA_I) +
                            (LENET_WEIGHT_W - LENET_WEIGHT_I);      // 16
static const int NARROW_SHIFT = ACC_FRAC - (LENET_DATA_W - LENET_DATA_I); // 11

/*
 * Quantize a Q16 integer accumulator value into data_t (12-bit signed).
 *
 * Bit-exact equivalent of the reference narrow():
 *   base = value >> 11; rem = value & 0x7ff;
 *   round-half-to-even on rem, then saturate to [-2048, 2047].
 *
 * Used under LENET_ACC_INT. When acc_t is the legacy ap_fixed the caller
 * simply writes data_t(acc) directly (see the templated helper below).
 */
static inline data_t quantize_acc(const ap_int<33> acc)
{
    // Arithmetic >> keeps sign; the low NARROW_SHIFT bits are the remainder.
    // After >>11, `base` is already the 12-bit RAW pattern of data_t
    // (the Q16 sum shifted to Q5), not a real-valued number.
    ap_int<33> base = acc >> NARROW_SHIFT;
    ap_uint<NARROW_SHIFT> rem = acc.range(NARROW_SHIFT - 1, 0);
    const ap_uint<NARROW_SHIFT> half = ap_uint<NARROW_SHIFT>(1) << (NARROW_SHIFT - 1);

    // Round half to even.
    bool round_up = (rem > half) || ((rem == half) && (base[0] == 1));
    if (round_up) base = base + 1;

    // Saturate to 12-bit signed raw range.
    ap_int<LENET_DATA_W> raw;
    if (base > 2047)      raw = 2047;
    else if (base < -2048) raw = -2048;
    else                  raw = base;

    // Pack raw bits directly (same as tests dfrom/rfrom convention).
    data_t r;
    r.range(LENET_DATA_W - 1, 0) = raw;
    return r;
}
#endif

template <int M, int N, int K>
void gemm_systolic(
    const data_t  A[M * K],
    const weight_t B[N * K],
    data_t         C[M * N])
{
#pragma HLS INLINE

    systolic_acc_t acc[M][N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            acc[i][j] = systolic_acc_t(0);
        }
    }

    for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
        data_t  a_loc[M];
        weight_t b_loc[N];
#pragma HLS ARRAY_PARTITION variable=a_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_loc complete dim=1
        // Load the M inputs and N weights exactly once per cycle.
        for (int i = 0; i < M; ++i) {
#pragma HLS UNROLL
            a_loc[i] = A[i * K + k];
        }
        for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
            b_loc[j] = B[j * K + k];
        }
        for (int i = 0; i < M; ++i) {
#pragma HLS UNROLL
            for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
                // Integer path: extract raw 12-bit bits of each operand, do a
                // pure signed multiply (24-bit) and integer add into acc. No
                // rounding, no saturation per MAC.
#if defined(LENET_USE_FIXED) && defined(LENET_ACC_INT)
                ap_int<LENET_DATA_W>   a_raw = a_loc[i].range(LENET_DATA_W - 1, 0);
                ap_int<LENET_WEIGHT_W> b_raw = b_loc[j].range(LENET_WEIGHT_W - 1, 0);
#if defined(LENET_FC_LUT)
                ap_int<LENET_DATA_W + LENET_WEIGHT_W> product =
                    (ap_int<LENET_DATA_W + LENET_WEIGHT_W>)(a_raw * b_raw);
#pragma HLS RESOURCE variable=product core=Mul_LUT
                acc[i][j] += product;
#else
                acc[i][j] += (ap_int<LENET_DATA_W + LENET_WEIGHT_W>)(a_raw * b_raw);
#endif
#else
                acc[i][j] += a_loc[i] * b_loc[j];
#endif
            }
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
#if defined(LENET_USE_FIXED) && defined(LENET_ACC_INT)
            C[i * N + j] = quantize_acc(acc[i][j]);
#else
            C[i * N + j] = data_t(acc[i][j]);
#endif
        }
    }
}

#endif // GEMM_SYSTOLIC_H
