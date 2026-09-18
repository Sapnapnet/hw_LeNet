#ifndef MEMBER5_FC_H
#define MEMBER5_FC_H

/*
 * Fully-connected layer, no bias: y[OC] = W[OC][IC] * x[IC].
 * Weight layout [OUT][IN] matches the frozen member1/member3 export and the
 * gemm_systolic B operand directly (no transpose).
 *
 * Reuses member4's output-stationary GEMM. In fixed mode the member3
 * narrowing (>>11 round-half-to-even + 12-bit saturation) is built into
 * gemm_systolic's write-back, so FC outputs are already data_t-aligned.
 */

#include "../conv/gemm_systolic.h"

// Full-parallel: all OC output channels in one output-stationary pass.
template <int OC, int IC>
void fc(const data_t in[IC], const weight_t w[OC * IC], data_t out[OC]) {
#pragma HLS INLINE
    gemm_systolic<1, OC, IC>(in, w, out);
}

// Tiled: N_TILE output channels per pass, to bound DSP/LUT usage.
// w is viewed as [OC/N_TILE][N_TILE*IC] (identical memory layout as the flat
// [OUT][IN] export); the dim-2 block partition gives each tile pass N_TILE
// clean per-channel banks without runtime address division.
template <int OC, int IC, int N_TILE>
void fc_tiled(const data_t in[IC], const weight_t w[][N_TILE * IC], data_t out[OC]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=w block factor=N_TILE dim=2
    static_assert(OC % N_TILE == 0, "OC must be a multiple of N_TILE");
    for (int t = 0; t < OC / N_TILE; ++t) {
        gemm_systolic<1, N_TILE, IC>(in, &w[t][0], &out[t * N_TILE]);
    }
}

#endif // MEMBER5_FC_H
