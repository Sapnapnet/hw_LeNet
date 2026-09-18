#include "../hls/top/lenet_accelerator.h"
#include "fast_kernels.h"

void lenet_accelerator_with_weights(
    const data_t input[INPUT_SIZE],
    const weight_t conv1_w[CONV1_WEIGHT_COUNT],
    const weight_t conv2_w[CONV2_WEIGHT_COUNT],
    const weight_t fc1_w[FC1_OUT/8][8*FC1_IN],
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
    data_t input_buffer[8][INPUT_SIZE/8];
    data_t conv1_buffer[2][CONV1_OUT_SIZE/2];
    data_t pool1_buffer[8][POOL1_OUT_SIZE/8];
    data_t conv2_buffer[2][CONV2_OUT_SIZE/2];
    data_t pool2_storage[1][POOL2_OUT_SIZE];
    data_t *pool2_buffer=pool2_storage[0];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=pool1_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv1_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv2_buffer complete dim=1
    for(unsigned i=0;i<INPUT_SIZE;++i) {
#pragma HLS PIPELINE II=1
        input_buffer[i%8][i/8]=input[i];
    }
    conv_banked<1,6,28,28,6>(input_buffer,conv1_w,conv1_buffer);
    relu_pool<6,24,24,8>(conv1_buffer,pool1_buffer);
    conv_banked<6,16,12,12,8>(pool1_buffer,conv2_w,conv2_buffer);
    relu_pool<16,8,8,1>(conv2_buffer,pool2_storage);
#endif

#ifdef LENET_STRAIGHT_ONLY
    data_t flatten_buffer[FLATTEN_SIZE];
    flatten<FLATTEN_C,FLATTEN_H,FLATTEN_W>(pool2_buffer,flatten_buffer);
    fc_tiled<FC1_OUT,FC1_IN,8>(flatten_buffer,fc1_w,fc1_buffer);
#else
    // Pool2 writes c*4*4+h*4+w, exactly the FC input order. No copy,
    // reinterpret cast, transpose or FC weight permutation is necessary.
    static_assert(POOL2_OUT_SIZE==FC1_IN, "pool2/FC1 shape mismatch");
    fc_tiled<FC1_OUT,FC1_IN,8>(pool2_buffer,fc1_w,fc1_buffer);
#endif
    relu<FC1_OUT>(fc1_buffer,relu3_buffer);
    fc_tiled<FC2_OUT,FC2_IN,12>(relu3_buffer,fc2_w,fc2_buffer);
    relu<FC2_OUT>(fc2_buffer,relu4_buffer);
    fc_tiled<FC3_OUT,FC3_IN,5>(relu4_buffer,fc3_w,output_buffer);
    m6_buffer_copy<NUM_CLASSES>(output_buffer,logits);
}
