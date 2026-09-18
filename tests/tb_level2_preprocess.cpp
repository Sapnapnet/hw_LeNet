#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "../hls/preprocess/level2_preprocess.h"

static void set_rgb(uint8_t image[], int width, int x, int y,
                    uint8_t red, uint8_t green, uint8_t blue) {
    const int index = 3 * (y * width + x);
    image[index] = red;
    image[index + 1] = green;
    image[index + 2] = blue;
}

static int count_white(const uint8_t image[LEVEL2_OUTPUT_PIXELS]) {
    int total = 0;
    for (int i = 0; i < LEVEL2_OUTPUT_PIXELS; ++i) total += image[i] == 255 ? 1 : 0;
    return total;
}

int main() {
    // Simple path: black/white RGB must remain black/white after grayscale + resize.
    uint8_t simple_source[4 * 4 * 3];
    std::memset(simple_source, 0, sizeof(simple_source));
    set_rgb(simple_source, 4, 3, 3, 255, 255, 255);
    uint8_t simple[LEVEL2_OUTPUT_PIXELS];
    assert(level2_simple_preprocess_rgb(simple_source, 4, 4, simple) == 0);
    assert(simple[0] == 0);
    assert(simple[LEVEL2_OUTPUT_PIXELS - 1] == 255);

    // Full path: dark ink on light paper is detected, cropped and centred.
    uint8_t digit[32 * 24 * 3];
    std::memset(digit, 255, sizeof(digit));
    for (int y = 5; y <= 18; ++y)
        for (int x = 11; x <= 14; ++x)
            set_rgb(digit, 32, x, y, 0, 0, 0);
    uint8_t full[LEVEL2_OUTPUT_PIXELS];
    Level2FullPreprocessInfo info;
    assert(level2_full_preprocess_rgb(digit, 32, 24, full, info) == 0);
    assert(info.valid_roi == 1);
    assert(info.foreground_pixels == 56);
    assert(info.roi_left == 11 && info.roi_right == 14);
    assert(info.roi_top == 5 && info.roi_bottom == 18);
    assert(count_white(full) > 0);
    assert(full[14 * LEVEL2_OUTPUT_SIDE + 14] == 255);

    // Full path must also support inverse contrast: light digit on dark paper.
    uint8_t inverse[16 * 16 * 3];
    std::memset(inverse, 0, sizeof(inverse));
    for (int y = 3; y <= 12; ++y)
        for (int x = 6; x <= 9; ++x)
            set_rgb(inverse, 16, x, y, 255, 255, 255);
    assert(level2_full_preprocess_rgb(inverse, 16, 16, full, info) == 0);
    assert(info.valid_roi == 1);
    assert(info.foreground_pixels == 40);
    assert(count_white(full) > 0);

    // A uniform background has no ROI and remains all-zero in the Full path.
    uint8_t background[10 * 10 * 3];
    std::memset(background, 200, sizeof(background));
    assert(level2_full_preprocess_rgb(background, 10, 10, full, info) == 0);
    assert(info.valid_roi == 0);
    assert(info.foreground_pixels == 0);
    assert(count_white(full) == 0);

    int16_t raw[LEVEL2_OUTPUT_PIXELS];
    full[0] = 0;
    full[1] = 255;
    full[2] = 128;
    level2_export_a12_raw(full, raw);
    assert(raw[0] == 0);
    assert(raw[1] == 32);
    assert(raw[2] == 16);

    std::cout << "PASS: Level 2 HLS preprocessing unit tests\n";
    return 0;
}
