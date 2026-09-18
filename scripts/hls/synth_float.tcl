# Vivado HLS 2018.3 C synthesis for the integrated Float top.
set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
file mkdir build results/synthesis/float
cd build
open_project -reset hls_synth_float
set_top lenet_accelerator_with_weights
add_files ../hls/conv/conv2d_systolic.cpp
add_files ../hls/top/lenet_accelerator.cpp
open_solution -reset solution1
set_part xc7z020clg400-1
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog -rtl verilog -rtl vhdl
exit
