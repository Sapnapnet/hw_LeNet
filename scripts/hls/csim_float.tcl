# Vivado HLS 2018.3 C Simulation entry point for the unified source tree.
# Invoke from the repository root with: vivado_hls.bat -f scripts/hls/csim_float.tcl

set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
file mkdir build results/accuracy
cd build
set image_file [file join $root data mnist raw t10k-images-idx3-ubyte]
set label_file [file join $root data mnist raw t10k-labels-idx1-ubyte]
set weight_dir [file join $root weights float]
set output_file [file join $root results accuracy hls_float_hls_csim_10000.csv]

open_project -reset hls_float
set_top lenet_accelerator_with_weights
add_files ../hls/conv/conv2d_systolic.cpp
add_files ../hls/top/lenet_accelerator.cpp
add_files -tb ../tests/tb_mnist_10k.cpp
open_solution -reset solution1
set_part xc7z020clg400-1
create_clock -period 10 -name default
csim_design -clean -argv [list --images $image_file --labels $label_file --weights $weight_dir --output $output_file --limit 10000]
exit
