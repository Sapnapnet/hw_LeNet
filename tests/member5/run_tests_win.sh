#!/usr/bin/env bash
# Windows Git Bash + MinGW local harness.
# The integrated test has canonical relative paths and is run from this
# directory so its reference paths resolve correctly.
set -euo pipefail
cd "$(dirname "$0")"
HLS_INC="${HLS_INC:-/d/Xilinx/Vivado/2018.3/include}"
mkdir -p ../../build
PKG="$(cd ../.. && pwd)"
cd "$PKG/tests/member5"
CXXFLAGS="-std=c++17 -O2 -static"
FDEF=''
XDEF=''

echo "== Member5 Float =="
g++ $CXXFLAGS $FDEF tb_member5.cpp -o ../../build/tb_member5_float.exe
../../build/tb_member5_float.exe

echo "== Member5 Fixed (integer acc) =="
g++ $CXXFLAGS -DLENET_USE_FIXED -DLENET_ACC_INT -I"$HLS_INC" $XDEF \
    tb_member5.cpp -o ../../build/tb_member5_fixed.exe
../../build/tb_member5_fixed.exe

echo "== Member5 Fixed (legacy ap_fixed acc) =="
g++ $CXXFLAGS -DLENET_USE_FIXED -I"$HLS_INC" $XDEF \
    tb_member5.cpp -o ../../build/tb_member5_legacy.exe
../../build/tb_member5_legacy.exe
