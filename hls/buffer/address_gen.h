#ifndef MEMBER6_ADDRESS_GEN_H
#define MEMBER6_ADDRESS_GEN_H

#include "../../config/network_config.h"

// feat CHW
inline int m6_chw_addr(int c, int h, int w, int H, int W) {
#pragma HLS INLINE
    return chw_index(c, h, w, H, W);
}

// conv [OUT][IN][KH][KW]
inline int m6_conv_weight_addr(int oc, int ic, int kh, int kw, int CIN, int KH, int KW) {
#pragma HLS INLINE
    return conv_weight_index(oc, ic, kh, kw, CIN, KH, KW);
}

// fc [OUT][IN]
inline int m6_fc_weight_addr(int out, int in, int IN) {
#pragma HLS INLINE
    return fc_weight_index(out, in, IN);
}

#endif
