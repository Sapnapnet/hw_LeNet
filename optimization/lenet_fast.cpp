#include "../hls/top/lenet_accelerator.h"
#include "fast_kernels.h"

void lenet_accelerator_with_weights(
    const data_t input[INPUT_SIZE],
    const weight_t conv1_w[CONV1_WEIGHT_COUNT],
#if defined(LENET_AGGRESSIVE)
    const weight_t conv2_w[CONV2_WEIGHT_COUNT],
#else
    const weight_t conv2_w[CONV2_WEIGHT_COUNT],
#endif
#if defined(LENET_FC1_TILE24)
    const weight_t fc1_w[FC1_OUT/24][24*FC1_IN],
#else
    const weight_t fc1_w[FC1_OUT/8][8*FC1_IN],
#endif
    const weight_t fc2_w[FC2_OUT/12][12*FC2_IN],
    const weight_t fc3_w[FC3_OUT/5][5*FC3_IN],
    output_t logits[NUM_CLASSES]) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
    data_t fc1_buffer[FC1_OUT], relu3_buffer[FC1_OUT];
    data_t fc2_buffer[FC2_OUT], relu4_buffer[FC2_OUT];
    data_t output_buffer[NUM_CLASSES];

#if defined(LENET_STRAIGHT_ONLY) || defined(LENET_NOFLATTEN_ONLY)
    data_t input_buffer[INPUT_SIZE];
    data_t conv1_buffer[CONV1_OUT_SIZE];
    data_t pool1_buffer[POOL1_OUT_SIZE];
    data_t conv2_buffer[CONV2_OUT_SIZE];
    data_t pool2_buffer[POOL2_OUT_SIZE];
    data_t relu1_buffer[CONV1_OUT_SIZE], relu2_buffer[CONV2_OUT_SIZE];
    m6_buffer_copy<INPUT_SIZE>(input,input_buffer);
    conv1_systolic(input_buffer,conv1_w,conv1_buffer);
    relu<CONV1_OUT_SIZE>(conv1_buffer,relu1_buffer);
    maxpool2d<POOL1_C,POOL1_IN_H,POOL1_IN_W>(relu1_buffer,pool1_buffer);
    conv2_systolic(pool1_buffer,conv2_w,conv2_buffer);
    relu<CONV2_OUT_SIZE>(conv2_buffer,relu2_buffer);
    maxpool2d<POOL2_C,POOL2_IN_H,POOL2_IN_W>(relu2_buffer,pool2_buffer);
#else
#if defined(LENET_AGGRESSIVE) || defined(LENET_AGGRESSIVE12) || defined(LENET_AGGRESSIVE5)
    data_t input_buffer[28][28];
#else
    data_t input_buffer[8][INPUT_SIZE/8];
#endif
    data_t conv1_buffer[2][CONV1_OUT_SIZE/2];
    data_t pool1_buffer[8][POOL1_OUT_SIZE/8];
    data_t conv2_buffer[2][CONV2_OUT_SIZE/2];
    data_t pool2_storage[1][POOL2_OUT_SIZE];
    data_t *pool2_buffer=pool2_storage[0];
#if defined(LENET_AGGRESSIVE) || defined(LENET_AGGRESSIVE12) || defined(LENET_AGGRESSIVE5)
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=0
#else
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
#endif
#pragma HLS ARRAY_PARTITION variable=pool1_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv1_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv2_buffer complete dim=1
    for(unsigned i=0;i<INPUT_SIZE;++i) {
#pragma HLS PIPELINE II=1
#if defined(LENET_AGGRESSIVE) || defined(LENET_AGGRESSIVE12) || defined(LENET_AGGRESSIVE5)
        input_buffer[i/28][i%28]=input[i];
#else
        input_buffer[i%8][i/8]=input[i];
#endif
    }
#if defined(LENET_AGGRESSIVE) || defined(LENET_AGGRESSIVE12) || defined(LENET_AGGRESSIVE5)
    conv1_banked_m16<CONV1_COUT>(input_buffer,conv1_w,conv1_buffer);
#else
    conv_banked<1,6,28,28,6>(input_buffer,conv1_w,conv1_buffer);
#endif
    relu_pool<CONV1_COUT,24,24,8>(conv1_buffer,pool1_buffer);
    // All 16 channels have a separate weight bank. 128 Conv2 MACs plus
    // 48 Conv1 and 25 FC MACs fit the device's 220 DSP budget.
#if defined(LENET_AGGRESSIVE5)
    conv_banked<5,12,12,12,12>(pool1_buffer,conv2_w,conv2_buffer);
    relu_pool<12,8,8,1>(conv2_buffer,pool2_storage);
#elif defined(LENET_AGGRESSIVE12)
    conv_banked<6,12,12,12,12>(pool1_buffer,conv2_w,conv2_buffer);
    relu_pool<12,8,8,1>(conv2_buffer,pool2_storage);
#elif defined(LENET_AGGRESSIVE)
    conv_banked<6,14,12,12,14>(pool1_buffer,conv2_w,conv2_buffer);
    relu_pool<14,8,8,1>(conv2_buffer,pool2_storage);
#else
    conv_banked<6,16,12,12,16>(pool1_buffer,conv2_w,conv2_buffer);
    relu_pool<16,8,8,1>(conv2_buffer,pool2_storage);
#endif
#endif

#ifdef LENET_STRAIGHT_ONLY
    data_t flatten_buffer[FLATTEN_SIZE];
    flatten<FLATTEN_C,FLATTEN_H,FLATTEN_W>(pool2_buffer,flatten_buffer);
    fc_tiled<FC1_OUT,FC1_IN,8>(flatten_buffer,fc1_w,fc1_buffer);
#else
    // Pool2 writes c*4*4+h*4+w, exactly the FC input order. No copy,
    // reinterpret cast, transpose or FC weight permutation is necessary.
    static_assert(POOL2_OUT_SIZE==FC1_IN, "pool2/FC1 shape mismatch");
#if defined(LENET_FC1_TILE24)
    fc_tiled<FC1_OUT,FC1_IN,24>(pool2_buffer,fc1_w,fc1_buffer);
#else
    fc_tiled<FC1_OUT,FC1_IN,8>(pool2_buffer,fc1_w,fc1_buffer);
#endif
#endif
    relu<FC1_OUT>(fc1_buffer,relu3_buffer);
    fc_tiled<FC2_OUT,FC2_IN,12>(relu3_buffer,fc2_w,fc2_buffer);
    relu<FC2_OUT>(fc2_buffer,relu4_buffer);
    fc_tiled<FC3_OUT,FC3_IN,5>(relu4_buffer,fc3_w,output_buffer);
    m6_buffer_copy<NUM_CLASSES>(output_buffer,logits);
}
