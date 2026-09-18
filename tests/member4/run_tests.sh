#!/usr/bin/env bash
# Local WSL tests for systolic Conv (GEMM reuse core).
# Requires a Vitis HLS include path for ap_fixed.h.
set -euo pipefail
cd "$(dirname "$0")"
HLS_INC="${HLS_INC:-/d/Xilinx/Vivado/2018.3/include}"

echo "== Conv Float =="
g++ -std=c++17 -O2 tb_conv_systolic.cpp ../../hls/conv/conv2d_systolic.cpp -o /tmp/tb_conv_systolic_float
/tmp/tb_conv_systolic_float

echo "== Conv Fixed (ap_fixed, integer acc - LUT optimized) =="
g++ -std=c++17 -O2 -DLENET_USE_FIXED -DLENET_ACC_INT -I"${HLS_INC}" \
    tb_conv_systolic.cpp ../../hls/conv/conv2d_systolic.cpp -o /tmp/tb_conv_systolic_fixed
/tmp/tb_conv_systolic_fixed

echo "== Conv Fixed (legacy ap_fixed acc - reference) =="
g++ -std=c++17 -O2 -DLENET_USE_FIXED -I"${HLS_INC}" \
    tb_conv_systolic.cpp ../../hls/conv/conv2d_systolic.cpp -o /tmp/tb_conv_systolic_legacy
/tmp/tb_conv_systolic_legacy
