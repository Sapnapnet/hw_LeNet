#pragma once
#include "../hls/conv/gemm_systolic.h"

// Eight adjacent x positions are in different cyclic input banks. Flatten K
// explicitly so the MAC pipeline does not drain at every 5-tap kernel row.
template<int CIN, int COUT, int H, int W, int NT>
void conv_banked(const data_t in[CIN*H*W],
                 const weight_t weights[COUT*CIN*25],
                 data_t out[COUT*(H-4)*(W-4)]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=weights block factor=NT dim=1
    const int OH=H-4, OW=W-4, K=CIN*25;
    static_assert(OW%8==0, "spatial tile must not cross output rows");
    for(int base=0; base<OH*OW; base+=8) {
        for(int oc=0; oc<COUT; oc+=NT) {
            systolic_acc_t acc[8][NT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            for(int m=0;m<8;++m)
                for(int j=0;j<NT;++j) {
#pragma HLS UNROLL
                    acc[m][j]=0;
                }
            mac_k: for(int k=0;k<K;++k) {
#pragma HLS PIPELINE II=1
                const int ic=k/25, kh=(k%25)/5, kw=k%5;
                data_t av[8];
                weight_t wv[NT];
#pragma HLS ARRAY_PARTITION variable=av complete
#pragma HLS ARRAY_PARTITION variable=wv complete
                for(int m=0;m<8;++m) {
#pragma HLS UNROLL
                    av[m]=in[(ic*H+base/OW+kh)*W+base%OW+m+kw];
                }
                for(int j=0;j<NT;++j) {
#pragma HLS UNROLL
                    wv[j]=weights[(oc+j)*K+k];
                }
                for(int m=0;m<8;++m) {
#pragma HLS UNROLL
                    for(int j=0;j<NT;++j) {
#pragma HLS UNROLL
                        ap_int<LENET_DATA_W> a=av[m].range(LENET_DATA_W-1,0);
                        ap_int<LENET_WEIGHT_W> b=wv[j].range(LENET_WEIGHT_W-1,0);
                        acc[m][j]+=(ap_int<LENET_DATA_W+LENET_WEIGHT_W>)(a*b);
                    }
                }
            }
            for(int j=0;j<NT;++j) {
                for(int m=0;m<8;++m) {
#pragma HLS PIPELINE II=1
                    out[(oc+j)*OH*OW+base+m]=quantize_acc(acc[m][j]);
                }
            }
        }
    }
}

// max(ReLU(a),...) == ReLU(max(a,...)); preserve quantization before pool.
template<int C,int H,int W>
void relu_pool(const data_t in[C*H*W], data_t out[C*H*W/4]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=2 dim=1
    for(int i=0;i<C*H*W/4;++i) {
#pragma HLS PIPELINE II=1
        const int c=i/((H/2)*(W/2));
        const int r=(i/ (W/2))%(H/2), col=i%(W/2);
        const int p=c*H*W+2*r*W+2*col;
        data_t a=in[p]>in[p+1]?in[p]:in[p+1];
        data_t b=in[p+W]>in[p+W+1]?in[p+W]:in[p+W+1];
        data_t v=a>b?a:b;
        out[i]=v>data_t(0)?v:data_t(0);
    }
}
