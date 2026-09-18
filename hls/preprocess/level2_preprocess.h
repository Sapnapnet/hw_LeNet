#ifndef LENET_LEVEL2_PREPROCESS_H
#define LENET_LEVEL2_PREPROCESS_H

#include <stdint.h>

// Input bounds cover the current Level 2 raw images (largest: 600 x 410).
// They are compile-time constants so Vivado HLS can infer bounded storage.
static const int LEVEL2_MAX_WIDTH = 640;
static const int LEVEL2_MAX_HEIGHT = 480;
static const int LEVEL2_MAX_PIXELS = LEVEL2_MAX_WIDTH * LEVEL2_MAX_HEIGHT;
static const int LEVEL2_OUTPUT_SIDE = 28;
static const int LEVEL2_OUTPUT_PIXELS = LEVEL2_OUTPUT_SIDE * LEVEL2_OUTPUT_SIDE;
static const int LEVEL2_CONTENT_SIDE = 20;

struct Level2FullPreprocessInfo {
    int valid_roi;
    int threshold;
    int foreground_pixels;
    int roi_left;
    int roi_top;
    int roi_right;
    int roi_bottom;
};

// RGB input is packed as R,G,B bytes.  Output is an MNIST-oriented image:
// white foreground (255), black background (0), row-major 28 x 28.
// Returns 0 on success and -1 when input dimensions exceed the HLS bounds.
int level2_simple_preprocess_rgb(const uint8_t rgb[], int src_width, int src_height,
                                 uint8_t output[LEVEL2_OUTPUT_PIXELS]);

// Full preprocessing performs grayscale conversion, contrast thresholding,
// automatic foreground polarity, ROI detection, aspect-ratio-preserving resize,
// and centered padding.  `info` is emitted for audit/PPT evidence.
int level2_full_preprocess_rgb(const uint8_t rgb[], int src_width, int src_height,
                               uint8_t output[LEVEL2_OUTPUT_PIXELS],
                               Level2FullPreprocessInfo &info);

// Match the established fixed MNIST input contract exactly:
// raw = round_ties_to_even((pixel / 255) * 32).  For uint8 input, the integer
// expression below is exact because pixel*32/255 can never be half-integral.
// The result is an A12 signed raw value (scale = 1/32), in [0, 32].
void level2_export_a12_raw(const uint8_t image[LEVEL2_OUTPUT_PIXELS],
                           int16_t raw[LEVEL2_OUTPUT_PIXELS]);

#endif  // LENET_LEVEL2_PREPROCESS_H
