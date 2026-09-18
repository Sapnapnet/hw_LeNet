#include "lenet_accelerator.h"

void lenet_accelerator_with_weights(
    const data_t input[INPUT_SIZE],
    const weight_t conv1_w[CONV1_WEIGHT_COUNT],
    const weight_t conv2_w[CONV2_WEIGHT_COUNT],
#if defined(LENET_FC1_TILE24)
    const weight_t fc1_w[FC1_OUT / 24][24 * FC1_IN],
#else
    const weight_t fc1_w[FC1_OUT / 8][8 * FC1_IN],
#endif
    const weight_t fc2_w[FC2_OUT / 12][12 * FC2_IN],
    const weight_t fc3_w[FC3_OUT / 5][5 * FC3_IN],
    output_t logits[NUM_CLASSES]) {
#pragma HLS INTERFACE ap_ctrl_hs port=return

    data_t input_buffer[INPUT_SIZE];
    data_t conv1_buffer[CONV1_OUT_SIZE];
    data_t relu1_buffer[CONV1_OUT_SIZE];
    data_t pool1_buffer[POOL1_OUT_SIZE];
    data_t conv2_buffer[CONV2_OUT_SIZE];
    data_t relu2_buffer[CONV2_OUT_SIZE];
    data_t pool2_buffer[POOL2_OUT_SIZE];
    data_t flatten_buffer[FLATTEN_SIZE];
    data_t fc1_buffer[FC1_OUT];
    data_t relu3_buffer[FC1_OUT];
    data_t fc2_buffer[FC2_OUT];
    data_t relu4_buffer[FC2_OUT];
    data_t output_buffer[NUM_CLASSES];

    member6_controller(
        input, conv1_w, conv2_w, fc1_w, fc2_w, fc3_w,
        input_buffer, conv1_buffer, relu1_buffer, pool1_buffer,
        conv2_buffer, relu2_buffer, pool2_buffer, flatten_buffer,
        fc1_buffer, relu3_buffer, fc2_buffer, relu4_buffer,
        output_buffer, logits);
}
