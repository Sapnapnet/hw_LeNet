#ifndef MEMBER6_BUFFER_H
#define MEMBER6_BUFFER_H

#include "address_gen.h"
#include "../../config/types.h"

// 线性 Buffer
template <typename T, int SIZE>
inline T m6_buffer_read(const T buffer[SIZE], int address) {
#pragma HLS INLINE
    return buffer[address];
}

template <typename T, int SIZE>
inline void m6_buffer_write(T buffer[SIZE], int address, T value) {
#pragma HLS INLINE
    buffer[address] = value;
}

template <int SIZE>
void m6_buffer_copy(const data_t source[SIZE], data_t destination[SIZE]) {
#pragma HLS INLINE
    for (int i = 0; i < SIZE; ++i) {
#pragma HLS PIPELINE II=1
        destination[i] = source[i];
    }
}

template <int C, int H, int W>
inline data_t m6_fmap_read(const data_t buffer[C * H * W], int c, int h, int w) {
#pragma HLS INLINE
    return m6_buffer_read<data_t, C * H * W>(
        buffer, m6_chw_addr(c, h, w, H, W));
}

template <int C, int H, int W>
inline void m6_fmap_write(data_t buffer[C * H * W], int c, int h, int w, data_t value) {
#pragma HLS INLINE
    m6_buffer_write<data_t, C * H * W>(
        buffer, m6_chw_addr(c, h, w, H, W), value);
}

#endif
