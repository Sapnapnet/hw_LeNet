// Member 6 address and minimal-buffer unit test.

#include <iostream>

#include "../hls/buffer/address_gen.h"
#include "../hls/buffer/buffer.h"

static bool expect_equal(const char *name, int got, int expected) {
    if (got != expected) {
        std::cerr << "FAIL: " << name << " got=" << got
                  << " expected=" << expected << "\n";
        return false;
    }
    return true;
}

int main() {
    bool ok = true;

    // 2x3x4 CHW: channel 0 occupies 0..11; channel 1 occupies 12..23.
    ok = expect_equal("CHW first", m6_chw_addr(0, 0, 0, 3, 4), 0) && ok;
    ok = expect_equal("CHW c0-last", m6_chw_addr(0, 2, 3, 3, 4), 11) && ok;
    ok = expect_equal("CHW c1-first", m6_chw_addr(1, 0, 0, 3, 4), 12) && ok;
    ok = expect_equal("CHW last", m6_chw_addr(1, 2, 3, 3, 4), 23) && ok;

    // OIHW [2][3][2][2].
    ok = expect_equal("OIHW first", m6_conv_weight_addr(
        0, 0, 0, 0, 3, 2, 2), 0) && ok;
    ok = expect_equal("OIHW oc0-last", m6_conv_weight_addr(
        0, 2, 1, 1, 3, 2, 2), 11) && ok;
    ok = expect_equal("OIHW oc1-first", m6_conv_weight_addr(
        1, 0, 0, 0, 3, 2, 2), 12) && ok;
    ok = expect_equal("OIHW last", m6_conv_weight_addr(
        1, 2, 1, 1, 3, 2, 2), 23) && ok;

    // FC [OUT][IN].
    ok = expect_equal("FC first", m6_fc_weight_addr(0, 0, 7), 0) && ok;
    ok = expect_equal("FC row boundary", m6_fc_weight_addr(1, 0, 7), 7) && ok;
    ok = expect_equal("FC last", m6_fc_weight_addr(3, 6, 7), 27) && ok;

    data_t buffer[2 * 3 * 4];
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 3; ++h) {
            for (int w = 0; w < 4; ++w) {
                // Keep values in the W12/A12 representable range in both
                // Float and Fixed builds.
                const int value = 20 * c + 3 * h + w;
                m6_fmap_write<2, 3, 4>(buffer, c, h, w, data_t(value));
            }
        }
    }

    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 3; ++h) {
            for (int w = 0; w < 4; ++w) {
                const int expected = 20 * c + 3 * h + w;
                const int got = (int)m6_fmap_read<2, 3, 4>(buffer, c, h, w);
                ok = expect_equal("CHW write/read", got, expected) && ok;
            }
        }
    }

    data_t copied[2 * 3 * 4];
    m6_buffer_copy<2 * 3 * 4>(buffer, copied);
    for (int i = 0; i < 2 * 3 * 4; ++i) {
        ok = expect_equal("buffer copy", (int)copied[i], (int)buffer[i]) && ok;
    }

    if (!ok) return 1;
    std::cout << "MEMBER6 ADDRESS/BUFFER PASS\n";
    return 0;
}
