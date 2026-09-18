# Vivado HLS 2018.3 C synthesis for the integrated W12/A12 Fixed top.
set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
file mkdir build results/synthesis/fixed
cd build
open_project -reset hls_synth_fixed
set_top lenet_accelerator_with_weights
set hls_include {F:/Vivado/Vivado/2018.3/include}
if {[info exists ::env(HLS_INCLUDE)] && $::env(HLS_INCLUDE) ne ""} { set hls_include $::env(HLS_INCLUDE) }
set cflags "-std=c++11 -DLENET_USE_FIXED -DLENET_ACC_INT -I$hls_include"
add_files -cflags $cflags ../hls/conv/conv2d_systolic.cpp
add_files -cflags $cflags ../hls/top/lenet_accelerator.cpp
open_solution -reset solution1
set_part xc7z020clg400-1
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog -rtl verilog -rtl vhdl
exit
