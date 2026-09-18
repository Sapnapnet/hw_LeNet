#include "level2_preprocess.h"

static uint8_t rgb_to_gray(const uint8_t red, const uint8_t green, const uint8_t blue) {
#pragma HLS INLINE
    // 1:2:1 RGB luma approximation.  It preserves any neutral black/white
    // handwriting exactly and uses only shifts/adds, not DSP multipliers.
    const unsigned int red_term = static_cast<unsigned int>(red);
    const unsigned int green_term = static_cast<unsigned int>(green);
    const unsigned int blue_term = static_cast<unsigned int>(blue);
    return static_cast<uint8_t>((red_term + green_term + green_term + blue_term + 2U) >> 2);
}

static void clear_output(uint8_t output[LEVEL2_OUTPUT_PIXELS]) {
    for (int i = 0; i < LEVEL2_OUTPUT_PIXELS; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = 0;
    }
}

int level2_simple_preprocess_rgb(const uint8_t rgb[], const int src_width, const int src_height,
                                 uint8_t output[LEVEL2_OUTPUT_PIXELS]) {
#pragma HLS INTERFACE m_axi port=rgb offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=src_width bundle=control
#pragma HLS INTERFACE s_axilite port=src_height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (src_width <= 0 || src_height <= 0 ||
        src_width > LEVEL2_MAX_WIDTH || src_height > LEVEL2_MAX_HEIGHT) {
        clear_output(output);
        return -1;
    }

    // Pixel-centre nearest-neighbour resize.  The Simple baseline deliberately
    // contains no inversion, thresholding, crop, or centering operation.
    for (int out_y = 0; out_y < LEVEL2_OUTPUT_SIDE; ++out_y) {
        const int in_y = ((2 * out_y + 1) * src_height) / (2 * LEVEL2_OUTPUT_SIDE);
        for (int out_x = 0; out_x < LEVEL2_OUTPUT_SIDE; ++out_x) {
#pragma HLS PIPELINE II=1
            const int in_x = ((2 * out_x + 1) * src_width) / (2 * LEVEL2_OUTPUT_SIDE);
            const int source_index = 3 * (in_y * src_width + in_x);
            output[out_y * LEVEL2_OUTPUT_SIDE + out_x] =
                rgb_to_gray(rgb[source_index], rgb[source_index + 1], rgb[source_index + 2]);
        }
    }
    return 0;
}

int level2_full_preprocess_rgb(const uint8_t rgb[], const int src_width, const int src_height,
                               uint8_t output[LEVEL2_OUTPUT_PIXELS],
                               Level2FullPreprocessInfo &info) {
#pragma HLS INTERFACE m_axi port=rgb offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=src_width bundle=control
#pragma HLS INTERFACE s_axilite port=src_height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    clear_output(output);
    info.valid_roi = 0;
    info.threshold = 0;
    info.foreground_pixels = 0;
    info.roi_left = 0;
    info.roi_top = 0;
    info.roi_right = -1;
    info.roi_bottom = -1;

    if (src_width <= 0 || src_height <= 0 ||
        src_width > LEVEL2_MAX_WIDTH || src_height > LEVEL2_MAX_HEIGHT) {
        return -1;
    }

    int minimum = 255;
    int maximum = 0;
    long border_sum = 0;
    int border_count = 0;
    for (int y = 0; y < src_height; ++y) {
        for (int x = 0; x < src_width; ++x) {
#pragma HLS PIPELINE II=1
            const int source_index = 3 * (y * src_width + x);
            const uint8_t value = rgb_to_gray(rgb[source_index], rgb[source_index + 1], rgb[source_index + 2]);
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
            if (x == 0 || y == 0 || x == src_width - 1 || y == src_height - 1) {
                border_sum += static_cast<int>(value);
                ++border_count;
            }
        }
    }

    // The midpoint makes this data path deterministic and avoids an external
    // floating-point/Otsu dependency.  Border brightness decides whether the
    // foreground is dark ink on light paper or light ink on dark paper.
    const int threshold = minimum + (maximum - minimum) / 2;
    const int border_mean = border_sum / border_count;
    const bool foreground_is_dark = border_mean >= threshold;
    info.threshold = threshold;

    int left = src_width;
    int top = src_height;
    int right = -1;
    int bottom = -1;
    int foreground_count = 0;
    for (int y = 0; y < src_height; ++y) {
        for (int x = 0; x < src_width; ++x) {
#pragma HLS PIPELINE II=1
            const int source_index = 3 * (y * src_width + x);
            const uint8_t value = rgb_to_gray(rgb[source_index], rgb[source_index + 1], rgb[source_index + 2]);
            const bool foreground = foreground_is_dark ? (value < threshold) : (value > threshold);
            if (foreground) {
                ++foreground_count;
                if (x < left) left = x;
                if (x > right) right = x;
                if (y < top) top = y;
                if (y > bottom) bottom = y;
            }
        }
    }
    info.foreground_pixels = foreground_count;
    if (foreground_count == 0) return 0;

    const int roi_width = right - left + 1;
    const int roi_height = bottom - top + 1;
    int resized_width;
    int resized_height;
    if (roi_width >= roi_height) {
        resized_width = LEVEL2_CONTENT_SIDE;
        resized_height = (roi_height * LEVEL2_CONTENT_SIDE + roi_width / 2) / roi_width;
    } else {
        resized_height = LEVEL2_CONTENT_SIDE;
        resized_width = (roi_width * LEVEL2_CONTENT_SIDE + roi_height / 2) / roi_height;
    }
    if (resized_width < 1) resized_width = 1;
    if (resized_height < 1) resized_height = 1;
    const int offset_x = (LEVEL2_OUTPUT_SIDE - resized_width) / 2;
    const int offset_y = (LEVEL2_OUTPUT_SIDE - resized_height) / 2;

    for (int dst_y = 0; dst_y < resized_height; ++dst_y) {
        int src_y = top + ((2 * dst_y + 1) * roi_height) / (2 * resized_height);
        if (src_y > bottom) src_y = bottom;
        for (int dst_x = 0; dst_x < resized_width; ++dst_x) {
#pragma HLS PIPELINE II=1
            int src_x = left + ((2 * dst_x + 1) * roi_width) / (2 * resized_width);
            if (src_x > right) src_x = right;
            const int source_index = 3 * (src_y * src_width + src_x);
            const uint8_t value = rgb_to_gray(rgb[source_index], rgb[source_index + 1], rgb[source_index + 2]);
            const bool foreground = foreground_is_dark ? (value < threshold) : (value > threshold);
            output[(offset_y + dst_y) * LEVEL2_OUTPUT_SIDE + offset_x + dst_x] = foreground ? 255 : 0;
        }
    }

    info.valid_roi = 1;
    info.roi_left = left;
    info.roi_top = top;
    info.roi_right = right;
    info.roi_bottom = bottom;
    return 0;
}

void level2_export_a12_raw(const uint8_t image[LEVEL2_OUTPUT_PIXELS],
                           int16_t raw[LEVEL2_OUTPUT_PIXELS]) {
    for (int i = 0; i < LEVEL2_OUTPUT_PIXELS; ++i) {
#pragma HLS PIPELINE II=1
        raw[i] = static_cast<int16_t>((static_cast<int>(image[i]) * 32 + 127) / 255);
    }
}
