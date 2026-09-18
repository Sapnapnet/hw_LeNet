// Member 4 systolic Conv/MAC testbench.
//
// Build modes:
//   Float mode  : g++ -std=c++17 -O2 tb_conv_systolic.cpp
//   Fixed mode  : g++ -std=c++17 -O2 -DLENET_USE_FIXED \
//                 -I<vitis_hls>/include tb_conv_systolic.cpp
//
// Fixed mode requires ap_fixed.h from the installed Vitis HLS include path.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../hls/conv/conv2d_systolic.h"

typedef std::int64_t Raw;
typedef std::vector<Raw> RawTensor;
typedef std::vector<double> FloatTensor;

static RawTensor load_raw(const std::string &path, std::size_t expected) {
    std::ifstream f(path.c_str());
    if (!f) throw std::runtime_error("Cannot open " + path);
    RawTensor out;
    Raw v;
    while (f >> v) out.push_back(v);
    if (out.size() != expected) {
        throw std::runtime_error("Size mismatch " + path + " expected " +
                                 std::to_string(expected) + " got " +
                                 std::to_string(out.size()));
    }
    return out;
}

static FloatTensor load_float(const std::string &path, std::size_t expected) {
    std::ifstream f(path.c_str());
    if (!f) throw std::runtime_error("Cannot open " + path);
    FloatTensor out;
    double v;
    while (f >> v) out.push_back(v);
    if (out.size() != expected) {
        throw std::runtime_error("Size mismatch " + path + " expected " +
                                 std::to_string(expected) + " got " +
                                 std::to_string(out.size()));
    }
    return out;
}

#ifdef LENET_USE_FIXED

static data_t dfrom(Raw raw) {
    data_t v;
    v.range(LENET_DATA_W - 1, 0) = ap_int<LENET_DATA_W>(raw);
    return v;
}

static weight_t wfrom(Raw raw) {
    weight_t v;
    v.range(LENET_WEIGHT_W - 1, 0) = ap_int<LENET_WEIGHT_W>(raw);
    return v;
}

static Raw rfrom(data_t v) {
    ap_int<LENET_DATA_W> r;
    r.range(LENET_DATA_W - 1, 0) = v.range(LENET_DATA_W - 1, 0);
    return r.to_int64();
}

#else

static data_t dfrom(double v) { return static_cast<data_t>(v); }
static weight_t wfrom(double v) { return static_cast<weight_t>(v); }

#endif

// ---------------------------------------------------------------------------

#ifdef LENET_USE_FIXED

static void check_fixed_samples() {
    std::cout << "[Fixed] running exact raw checks on 5 real samples...\n";
    std::size_t total = 0;

    RawTensor rw1 = load_raw("../../weights/fixed/conv1.weight.txt", CONV1_WEIGHT_COUNT);
    RawTensor rw2 = load_raw("../../weights/fixed/conv2.weight.txt", CONV2_WEIGHT_COUNT);
    std::vector<data_t> dw1(rw1.size()), dw2(rw2.size());
    std::vector<weight_t> ww1(rw1.size()), ww2(rw2.size());
    for (std::size_t i = 0; i < rw1.size(); ++i) { dw1[i] = dfrom(rw1[i]); ww1[i] = wfrom(rw1[i]); }
    for (std::size_t i = 0; i < rw2.size(); ++i) { dw2[i] = dfrom(rw2[i]); ww2[i] = wfrom(rw2[i]); }

    for (int s = 0; s < 5; ++s) {
        std::ostringstream dir;
        dir << "../../reference/fixed/sample_" << std::setw(5) << std::setfill('0') << s << "/";

        RawTensor in1 = load_raw(dir.str() + "input.txt", INPUT_SIZE);
        RawTensor pool1 = load_raw(dir.str() + "pool1.txt", POOL1_OUT_SIZE);
        RawTensor exp1 = load_raw(dir.str() + "conv1.txt", CONV1_OUT_SIZE);
        RawTensor exp2 = load_raw(dir.str() + "conv2.txt", CONV2_OUT_SIZE);

        std::vector<data_t> din1(in1.size()), dpool1(pool1.size());
        for (std::size_t i = 0; i < in1.size(); ++i) din1[i] = dfrom(in1[i]);
        for (std::size_t i = 0; i < pool1.size(); ++i) dpool1[i] = dfrom(pool1[i]);

        std::vector<data_t> act1(CONV1_OUT_SIZE), act2(CONV2_OUT_SIZE);
        conv1_systolic(din1.data(), ww1.data(), act1.data());
        conv2_systolic(dpool1.data(), ww2.data(), act2.data());

        std::size_t mism1 = 0, mism2 = 0;
        Raw maxerr1 = 0, maxerr2 = 0;
        for (std::size_t i = 0; i < exp1.size(); ++i) {
            Raw got = rfrom(act1[i]);
            if (got != exp1[i]) ++mism1;
            maxerr1 = std::max(maxerr1, (Raw)std::llabs((long long)got - (long long)exp1[i]));
        }
        for (std::size_t i = 0; i < exp2.size(); ++i) {
            Raw got = rfrom(act2[i]);
            if (got != exp2[i]) ++mism2;
            maxerr2 = std::max(maxerr2, (Raw)std::llabs((long long)got - (long long)exp2[i]));
        }
        total += exp1.size() + exp2.size();
        std::cout << "  sample " << s << ": conv1 mism=" << mism1
                  << " conv2 mism=" << mism2
                  << " (maxerr " << maxerr1 << ", " << maxerr2 << ")\n";
        if (mism1 || mism2) throw std::runtime_error("Fixed mismatch in sample " + std::to_string(s));
    }
    std::cout << "[Fixed] samples exact, values=" << total << "\n";
}

static void check_fixed_generated() {
    std::cout << "[Fixed] running exact raw checks on generated vectors...\n";

    // artificial_small: CIN=2, COUT=1, H=4, W=4, K=3
    {
        const std::size_t CIN = 2, COUT = 1, H = 4, W = 4, K = 3;
        const std::size_t N = CIN * K * K, OH = H - K + 1, OW = W - K + 1, OSZ = COUT * OH * OW;
        RawTensor rin = load_raw("../../tests/vectors/artificial_small/input.txt", CIN * H * W);
        RawTensor rw = load_raw("../../tests/vectors/artificial_small/weight.txt", COUT * N);
        RawTensor exp = load_raw("../../tests/vectors/artificial_small/expected.txt", OSZ);
        std::vector<data_t> din(rin.size()), dw(rw.size());
        std::vector<weight_t> ww(rw.size());
        for (std::size_t i = 0; i < rin.size(); ++i) din[i] = dfrom(rin[i]);
        for (std::size_t i = 0; i < rw.size(); ++i) { dw[i] = dfrom(rw[i]); ww[i] = wfrom(rw[i]); }
        std::vector<data_t> act(OSZ);
        conv2d_gemm_direct<4, 1, 2, 1, 4, 4, 3, 3>(din.data(), ww.data(), act.data());
        std::size_t mism = 0;
        for (std::size_t i = 0; i < exp.size(); ++i) if (rfrom(act[i]) != exp[i]) ++mism;
        std::cout << "  artificial_small mism=" << mism << "\n";
        if (mism) throw std::runtime_error("artificial_small mismatch");
    }

    // random_conv1 / unit_impulse_conv1: CIN=1, COUT=6, H=28, W=28, K=5
    for (const char *name : {"random_conv1", "unit_impulse_conv1"}) {
        const std::size_t CIN = 1, COUT = 6, H = 28, W = 28, K = 5;
        const std::size_t N = CIN * K * K, OH = H - K + 1, OW = W - K + 1, OSZ = COUT * OH * OW;
        std::string base = std::string("../../tests/vectors/") + name;
        RawTensor rin = load_raw(base + "/input.txt", CIN * H * W);
        RawTensor rw = load_raw(base + "/weight.txt", COUT * N);
        RawTensor exp = load_raw(base + "/expected.txt", OSZ);
        std::vector<data_t> din(rin.size()), dw(rw.size());
        std::vector<weight_t> ww(rw.size());
        for (std::size_t i = 0; i < rin.size(); ++i) din[i] = dfrom(rin[i]);
        for (std::size_t i = 0; i < rw.size(); ++i) { dw[i] = dfrom(rw[i]); ww[i] = wfrom(rw[i]); }
        std::vector<data_t> act(OSZ);
        conv2d_gemm_direct<8, 6, 1, 6, 28, 28, 5, 5>(din.data(), ww.data(), act.data());
        std::size_t mism = 0;
        for (std::size_t i = 0; i < exp.size(); ++i) if (rfrom(act[i]) != exp[i]) ++mism;
        std::cout << "  " << name << " mism=" << mism << "\n";
        if (mism) throw std::runtime_error(std::string(name) + " mismatch");
    }
}

#else

static void check_float_samples() {
    std::cout << "[Float] running float checks on 5 real samples...\n";
    FloatTensor fw1 = load_float("../../weights/float/conv1.weight.txt", CONV1_WEIGHT_COUNT);
    FloatTensor fw2 = load_float("../../weights/float/conv2.weight.txt", CONV2_WEIGHT_COUNT);
    std::vector<data_t> w1(fw1.size()), w2(fw2.size());
    std::vector<weight_t> ww1(fw1.size()), ww2(fw2.size());
    for (std::size_t i = 0; i < fw1.size(); ++i) { w1[i] = dfrom(fw1[i]); ww1[i] = wfrom(fw1[i]); }
    for (std::size_t i = 0; i < fw2.size(); ++i) { w2[i] = dfrom(fw2[i]); ww2[i] = wfrom(fw2[i]); }

    double max_conv1 = 0, max_conv2 = 0;
    double sum_conv1 = 0, sum_conv2 = 0;
    std::size_t n_conv1 = 0, n_conv2 = 0;
    for (int s = 0; s < 5; ++s) {
        std::ostringstream dir;
        dir << "../../reference/float/sample_" << std::setw(5) << std::setfill('0') << s << "/";
        FloatTensor in1 = load_float(dir.str() + "input.txt", INPUT_SIZE);
        FloatTensor pool1 = load_float(dir.str() + "pool1.txt", POOL1_OUT_SIZE);
        FloatTensor exp1 = load_float(dir.str() + "conv1.txt", CONV1_OUT_SIZE);
        FloatTensor exp2 = load_float(dir.str() + "conv2.txt", CONV2_OUT_SIZE);

        std::vector<data_t> din1(in1.size()), dpool1(pool1.size());
        for (std::size_t i = 0; i < in1.size(); ++i) din1[i] = dfrom(in1[i]);
        for (std::size_t i = 0; i < pool1.size(); ++i) dpool1[i] = dfrom(pool1[i]);

        std::vector<data_t> act1(CONV1_OUT_SIZE), act2(CONV2_OUT_SIZE);
        conv1_systolic(din1.data(), ww1.data(), act1.data());
        conv2_systolic(dpool1.data(), ww2.data(), act2.data());

        for (std::size_t i = 0; i < exp1.size(); ++i) {
            double e = std::abs((double)act1[i] - exp1[i]);
            max_conv1 = std::max(max_conv1, e);
            sum_conv1 += e; ++n_conv1;
        }
        for (std::size_t i = 0; i < exp2.size(); ++i) {
            double e = std::abs((double)act2[i] - exp2[i]);
            max_conv2 = std::max(max_conv2, e);
            sum_conv2 += e; ++n_conv2;
        }
        std::cout << "  sample " << s << " done\n";
    }
    std::cout << "[Float] max abs error conv1=" << max_conv1
              << " conv2=" << max_conv2 << "\n";
    std::cout << "[Float] mean abs error conv1="
              << (n_conv1 ? sum_conv1 / n_conv1 : 0.0)
              << " conv2=" << (n_conv2 ? sum_conv2 / n_conv2 : 0.0) << "\n";
    if (max_conv1 > 1e-3 || max_conv2 > 1e-3) {
        throw std::runtime_error("Float error too large");
    }
}

#endif

int main() {
    try {
#ifdef LENET_USE_FIXED
        check_fixed_samples();
        check_fixed_generated();
        std::cout << "SYSTOLIC FIXED PASS\n";
#else
        check_float_samples();
        std::cout << "SYSTOLIC FLOAT PASS\n";
#endif
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
