#pragma once
#include "../hls/conv/gemm_systolic.h"

// Explicit banks are required for HLS 2018.3: cyclic partition of a linear
// array alone does not prove that dynamic consecutive addresses are disjoint.
template<int CIN, int COUT, int H, int W, int TILE_N>
void conv_banked(const data_t in[8][CIN*H*W/8],
                 const weight_t weights[COUT*CIN*25],
                 data_t out[2][COUT*(H-4)*(W-4)/2]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights block factor=TILE_N dim=1
    const int OH=H-4, OW=W-4, K=CIN*25;
    static_assert(OW%8==0, "spatial tile must not cross output rows");
    for(unsigned oh=0; oh<OH; ++oh) {
      for(unsigned ox=0; ox<OW; ox+=8) {
        for(int oc=0; oc<COUT; oc+=TILE_N) {
            systolic_acc_t acc[8][TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            for(int m=0;m<8;++m)
                for(int j=0;j<TILE_N;++j) {
#pragma HLS UNROLL
                    acc[m][j]=0;
                }
            for(unsigned ic=0;ic<CIN;++ic) {
              for(unsigned kh=0;kh<5;++kh) {
               mac_k: for(unsigned kw=0;kw<5;++kw) {
#pragma HLS PIPELINE II=1
                const unsigned k=(ic*5+kh)*5+kw;
                const unsigned start=(ic*H+oh+kh)*W+ox+kw;
                const unsigned offset=start%8;
                data_t av[8], bank_values[8];
                weight_t wv[TILE_N];
#pragma HLS ARRAY_PARTITION variable=av complete
#pragma HLS ARRAY_PARTITION variable=bank_values complete
#pragma HLS ARRAY_PARTITION variable=wv complete
                for(int bank=0;bank<8;++bank) {
#pragma HLS UNROLL
                    bank_values[bank]=in[bank][start/8+(bank<offset)];
                }
                for(int m=0;m<8;++m) {
#pragma HLS UNROLL
                    av[m]=bank_values[(offset+m)%8];
                }
                for(int j=0;j<TILE_N;++j) {
#pragma HLS UNROLL
                    wv[j]=weights[(oc+j)*K+k];
                }
                for(int m=0;m<8;++m) {
#pragma HLS UNROLL
                    for(int j=0;j<TILE_N;++j) {
#pragma HLS UNROLL
                        ap_int<LENET_DATA_W> a=av[m].range(LENET_DATA_W-1,0);
                        ap_int<LENET_WEIGHT_W> b=wv[j].range(LENET_WEIGHT_W-1,0);
                        acc[m][j]+=(ap_int<LENET_DATA_W+LENET_WEIGHT_W>)(a*b);
                    }
                }
               }
              }
            }
            for(int j=0;j<TILE_N;++j) {
                for(int m=0;m<8;++m) {
#pragma HLS PIPELINE II=1
                    const unsigned pos=(oc+j)*OH*OW+oh*OW+ox+m;
                    out[pos%2][pos/2]=quantize_acc(acc[m][j]);
                }
            }
        }
      }
    }
}

// max(ReLU(a),...) == ReLU(max(a,...)); preserve quantization before pool.
template<int C,int H,int W,int OB>
void relu_pool(const data_t in[2][C*H*W/2], data_t out[OB][C*H*W/4/OB]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS RESOURCE variable=in core=RAM_2P_BRAM
    for(int i=0;i<C*H*W/4;++i) {
#pragma HLS PIPELINE II=1
        const int c=i/((H/2)*(W/2));
        const int r=(i/ (W/2))%(H/2), col=i%(W/2);
        const int p=c*H*W+2*r*W+2*col;
        data_t a=in[0][p/2]>in[1][p/2]?in[0][p/2]:in[1][p/2];
        data_t b=in[0][(p+W)/2]>in[1][(p+W)/2]?in[0][(p+W)/2]:in[1][(p+W)/2];
        data_t v=a>b?a:b;
        out[i%OB][i/OB]=v>data_t(0)?v:data_t(0);
    }
}
