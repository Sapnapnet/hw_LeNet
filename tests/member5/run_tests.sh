#!/usr/bin/env bash
# Local WSL tests for member5 ReLU/MaxPool/Flatten/FC/Argmax.
# Fixed modes require an HLS include path for ap_fixed.h.
# On Windows Git Bash use run_tests_win.sh instead (MinGW cannot open
# Chinese paths).
set -euo pipefail
cd "$(dirname "$0")"
HLS_INC="${HLS_INC:-/mnt/d/Applications/Xilinx/Vitis_HLS/2020.2/include}"

echo "== Member5 Float =="
g++ -std=c++17 -O2 tb_member5.cpp -o /tmp/tb_member5_float
/tmp/tb_member5_float

echo "== Member5 Fixed (integer acc) =="
g++ -std=c++17 -O2 -DLENET_USE_FIXED -DLENET_ACC_INT -I"${HLS_INC}" \
    tb_member5.cpp -o /tmp/tb_member5_fixed
/tmp/tb_member5_fixed

echo "== Member5 Fixed (legacy ap_fixed acc) =="
g++ -std=c++17 -O2 -DLENET_USE_FIXED -I"${HLS_INC}" \
    tb_member5.cpp -o /tmp/tb_member5_legacy
/tmp/tb_member5_legacy
