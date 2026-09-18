open_project -reset level2_preprocess_csim
set_top level2_full_preprocess_rgb
add_files ../../hls/preprocess/level2_preprocess.cpp -cflags "-std=c++11"
add_files -tb ../../tests/tb_level2_preprocess.cpp -cflags "-std=c++11"
open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
exit
