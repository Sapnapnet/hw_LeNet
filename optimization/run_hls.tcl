set root [file normalize [file join [file dirname [info script]] ..]]
set variant baseline
if {[info exists ::env(LENET_VARIANT)]} {set variant $::env(LENET_VARIANT)}
set stage synth
if {[info exists ::env(LENET_STAGE)]} {set stage $::env(LENET_STAGE)}
set limit 3
if {[info exists ::env(LENET_SAMPLES)]} {set limit $::env(LENET_SAMPLES)}
cd $root
file mkdir optimization/results/$variant optimization/build
set flags "-std=c++11 -DLENET_USE_FIXED -DLENET_ACC_INT"
cd $root/optimization/build
set project $variant
if {$stage eq "csim"} {set project ${variant}_csim}
open_project $project
set_top lenet_accelerator_with_weights
if {$variant eq "baseline"} {
    add_files -cflags $flags $root/hls/conv/conv2d_systolic.cpp
    add_files -cflags $flags $root/hls/top/lenet_accelerator.cpp
} else {
    if {$variant eq "straight"} {append flags " -DLENET_STRAIGHT_ONLY"}
    if {$variant eq "noflatten"} {append flags " -DLENET_NOFLATTEN_ONLY"}
    if {$variant eq "fc24" || $variant eq "fc24b"} {append flags " -DLENET_FC1_TILE24"}
    if {$variant eq "aggressive"} {append flags " -DLENET_AGGRESSIVE -DLENET_FC1_TILE24 -DLENET_FC_LUT"}
    if {$variant eq "aggressive12" || $variant eq "aggressive12b"} {append flags " -DLENET_AGGRESSIVE12"}
    if {$variant eq "aggressive5" || $variant eq "aggressive5b"} {append flags " -DLENET_AGGRESSIVE5 -DLENET_FC1_TILE24"}
    add_files -cflags $flags $root/hls/conv/conv2d_systolic.cpp
    add_files -cflags $flags $root/optimization/lenet_fast.cpp
}
add_files -tb -cflags $flags $root/tests/tb_mnist_10k.cpp
open_solution solution1
set_part xc7z020clg400-1
create_clock -period 10 -name default
set args [list --images $root/data/mnist/raw/t10k-images-idx3-ubyte --labels $root/data/mnist/raw/t10k-labels-idx1-ubyte --weights $root/weights/fixed --output $root/optimization/results/$variant/${stage}_${limit}.csv --limit $limit]
if {$stage eq "synth"} {
    csynth_design
} elseif {$stage eq "csim"} {
    csim_design -clean -O -argv $args
} elseif {$stage eq "cosim"} {
    cosim_design -O -rtl verilog -tool xsim -trace_level none -argv $args
} else {
    error "Unknown stage: $stage"
}
exit
