#ifndef MEMBER6_CONTROLLER_H
#define MEMBER6_CONTROLLER_H

#include "../buffer/buffer.h"
#include "../../config/network_config.h"
#include "../conv/conv2d_systolic.h"
#include "../operators/relu.h"
#include "../operators/maxpool2d.h"
#include "../operators/flatten.h"
#include "../operators/fc.h"

enum member6_state_t {
    M6_LOAD_INPUT = 0,
    M6_CONV1,
    M6_RELU1,
    M6_POOL1,
    M6_CONV2,
    M6_RELU2,
    M6_POOL2,
    M6_FLATTEN,
    M6_FC1,
    M6_RELU3,
    M6_FC2,
    M6_RELU4,
    M6_FC3,
    M6_WRITE_OUTPUT,
    M6_DONE
};

inline void member6_controller(
    const data_t input[INPUT_SIZE],
    const weight_t conv1_w[CONV1_WEIGHT_COUNT],
    const weight_t conv2_w[CONV2_WEIGHT_COUNT],
    const weight_t fc1_w[FC1_OUT / 8][8 * FC1_IN],
    const weight_t fc2_w[FC2_OUT / 12][12 * FC2_IN],
    const weight_t fc3_w[FC3_OUT / 5][5 * FC3_IN],
    data_t input_buffer[INPUT_SIZE],
    data_t conv1_buffer[CONV1_OUT_SIZE],
    data_t relu1_buffer[CONV1_OUT_SIZE],
    data_t pool1_buffer[POOL1_OUT_SIZE],
    data_t conv2_buffer[CONV2_OUT_SIZE],
    data_t relu2_buffer[CONV2_OUT_SIZE],
    data_t pool2_buffer[POOL2_OUT_SIZE],
    data_t flatten_buffer[FLATTEN_SIZE],
    data_t fc1_buffer[FC1_OUT],
    data_t relu3_buffer[FC1_OUT],
    data_t fc2_buffer[FC2_OUT],
    data_t relu4_buffer[FC2_OUT],
    data_t output_buffer[NUM_CLASSES],
    data_t logits[NUM_CLASSES]) {
#pragma HLS INLINE off

    member6_state_t state = M6_LOAD_INPUT;
    while (state != M6_DONE) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=14
        switch (state) {
        case M6_LOAD_INPUT:
            m6_buffer_copy<INPUT_SIZE>(input, input_buffer);
            state = M6_CONV1;
            break;

        case M6_CONV1:
            conv1_systolic(input_buffer, conv1_w, conv1_buffer);
            state = M6_RELU1;
            break;

        case M6_RELU1:
            relu<CONV1_OUT_SIZE>(conv1_buffer, relu1_buffer);
            state = M6_POOL1;
            break;

        case M6_POOL1:
            maxpool2d<POOL1_C, POOL1_IN_H, POOL1_IN_W>(
                relu1_buffer, pool1_buffer);
            state = M6_CONV2;
            break;

        case M6_CONV2:
            conv2_systolic(pool1_buffer, conv2_w, conv2_buffer);
            state = M6_RELU2;
            break;

        case M6_RELU2:
            relu<CONV2_OUT_SIZE>(conv2_buffer, relu2_buffer);
            state = M6_POOL2;
            break;

        case M6_POOL2:
            maxpool2d<POOL2_C, POOL2_IN_H, POOL2_IN_W>(
                relu2_buffer, pool2_buffer);
            state = M6_FLATTEN;
            break;

        case M6_FLATTEN:
            flatten<FLATTEN_C, FLATTEN_H, FLATTEN_W>(
                pool2_buffer, flatten_buffer);
            state = M6_FC1;
            break;

        case M6_FC1: // [OUT][IN] hang first
            fc_tiled<FC1_OUT, FC1_IN, 8>(
                flatten_buffer, fc1_w, fc1_buffer);
            state = M6_RELU3;
            break;

        case M6_RELU3:
            relu<FC1_OUT>(fc1_buffer, relu3_buffer);
            state = M6_FC2;
            break;

        case M6_FC2:
            fc_tiled<FC2_OUT, FC2_IN, 12>(
                relu3_buffer, fc2_w, fc2_buffer);
            state = M6_RELU4;
            break;

        case M6_RELU4:
            relu<FC2_OUT>(fc2_buffer, relu4_buffer);
            state = M6_FC3;
            break;

        case M6_FC3:
            fc_tiled<FC3_OUT, FC3_IN, 5>(
                relu4_buffer, fc3_w, output_buffer);
            state = M6_WRITE_OUTPUT;
            break;

        case M6_WRITE_OUTPUT:
            m6_buffer_copy<NUM_CLASSES>(output_buffer, logits);
            state = M6_DONE;
            break;

        case M6_DONE:
        default:
            state = M6_DONE;
            break;
        }
    }
}

#endif
