# Vivado HLS 2018.3 Fixed C Simulation entry point for the unified source tree.
# Invoke from the repository root with: vivado_hls.bat -f scripts/hls/csim_fixed.tcl

set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
file mkdir build results/accuracy
cd build
set image_file [file join $root data mnist raw t10k-images-idx3-ubyte]
set label_file [file join $root data mnist raw t10k-labels-idx1-ubyte]
set weight_dir [file join $root weights fixed]
set output_file [file join $root results accuracy hls_fixed_hls_csim_10000.csv]
set hls_include {F:/Vivado/Vivado/2018.3/include}
if {[info exists ::env(HLS_INCLUDE)] && $::env(HLS_INCLUDE) ne ""} {
    set hls_include $::env(HLS_INCLUDE)
}
file mkdir results/accuracy

open_project -reset hls_fixed
set_top lenet_accelerator_with_weights
set cflags "-std=c++11 -DLENET_USE_FIXED -DLENET_ACC_INT -I$hls_include"
add_files -cflags $cflags ../hls/conv/conv2d_systolic.cpp
add_files -cflags $cflags ../hls/top/lenet_accelerator.cpp
add_files -tb -cflags $cflags ../tests/tb_mnist_10k.cpp
open_solution -reset solution1
set_part xc7z020clg400-1
create_clock -period 10 -name default
csim_design -clean -argv [list --images $image_file --labels $label_file --weights $weight_dir --output $output_file --limit 10000]
exit
