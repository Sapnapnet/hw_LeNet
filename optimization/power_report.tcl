set root [file normalize [file join [file dirname [info script]] ..]]
set variant [lindex $argv 0]
if {$variant ni {baseline optimized}} {error "expected baseline or optimized"}
set result $root/optimization/results/$variant/power
cd $result
open_checkpoint routed.dcp
set_operating_conditions -ambient_temp 25
read_saif -strip_path apatb_lenet_accelerator_with_weights_top/AESL_inst_lenet_accelerator_with_weights -out_file saif_unmatched.rpt $root/optimization/results/$variant/activity.saif
report_power -file activity_power.rpt
exit
