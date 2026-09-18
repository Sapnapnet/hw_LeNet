#ifndef MEMBER6_LENET_ACCELERATOR_H
#define MEMBER6_LENET_ACCELERATOR_H

#include "controller.h"

// 权重暂时从端口输入 先独立验证连接
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
    output_t logits[NUM_CLASSES]);

#endif
