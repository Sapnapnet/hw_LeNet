# Out-of-context accelerator core only; external weight memories/PS/DDR excluded.
# Usage: vivado -mode batch -source power_impl.tcl -tclargs baseline|optimized|aggressive5b
set root [file normalize [file join [file dirname [info script]] ..]]
set variant [lindex $argv 0]
if {$variant ni {baseline optimized aggressive5b}} {error "expected baseline, optimized or aggressive5b"}
set result $root/optimization/results/$variant/power
file mkdir $result
cd $result
set_param general.maxThreads 4
set rtl $root/optimization/build/$variant/solution1/syn/verilog
read_verilog [glob $rtl/*.v]
synth_design -top lenet_accelerator_with_weights -part xc7z020clg400-1 -mode out_of_context
create_clock -name ap_clk -period 10 [get_ports ap_clk]
set_input_delay 0 -clock ap_clk [get_ports -filter {DIRECTION == IN && NAME != ap_clk}]
set_output_delay 0 -clock ap_clk [get_ports -filter {DIRECTION == OUT}]
set_false_path -from [get_ports ap_rst]
write_checkpoint -force synthesized.dcp
opt_design
place_design
route_design
write_checkpoint -force routed.dcp
report_timing_summary -file timing.rpt
report_utilization -file utilization.rpt
set_operating_conditions -ambient_temp 25
report_power -file vectorless_power.rpt
if {[file exists $root/optimization/results/$variant/activity.saif]} {
    read_saif -strip_path apatb_lenet_accelerator_with_weights_top/AESL_inst_lenet_accelerator_with_weights -out_file saif_unmatched.rpt $root/optimization/results/$variant/activity.saif
    report_power -file activity_power.rpt
}
exit
