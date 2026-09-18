// Member 5 ReLU / MaxPool / Flatten / FC / Argmax testbench.
//
// Build modes:
//   Float : g++ -std=c++17 -O2 tb_member5.cpp
//   Fixed : g++ -std=c++17 -O2 -DLENET_USE_FIXED [-DLENET_ACC_INT] \
//           -I<hls_include> tb_member5.cpp
//
// Reference roots default to the package layout (WSL/Linux). Override with
// -DM5_REF_FLOAT_DIR=... / -DM5_REF_FIXED_DIR=... / -DM5_DATA_DIR=... on
// other setups.
//
// Alignment sources:
//   * relu1/pool1 : member2 float / member3 fixed raw official references
//   * relu2/pool2/flatten/fc1/relu3/fc2/relu4/logits : member5 data/
//     (real FC weights from lenet_float.pt, member3-rule fixed raw;
//     see data/PROVENANCE.md)

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

#include "../../hls/operators/relu.h"
#include "../../hls/operators/maxpool2d.h"
#include "../../hls/operators/flatten.h"
#include "../../hls/operators/fc.h"
#include "../../hls/operators/argmax10.h"
#include "../../config/network_config.h"

#ifndef M5_REF_FLOAT_DIR
#define M5_REF_FLOAT_DIR "../../reference/float"
#endif
#ifndef M5_REF_FIXED_DIR
#define M5_REF_FIXED_DIR "../../reference/fixed"
#endif
#ifndef M5_DATA_DIR
#define M5_DATA_DIR "../../"
#endif

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

static std::string sample_dir(const char *root, int s) {
    std::ostringstream d;
    d << root << "/sample_" << std::setw(5) << std::setfill('0') << s << "/";
    return d.str();
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

static std::size_t raw_mismatch(const std::vector<data_t> &a, const RawTensor &b) {
    std::size_t m = 0;
    for (std::size_t i = 0; i < b.size(); ++i) if (rfrom(a[i]) != b[i]) ++m;
    return m;
}

#else

static data_t dfrom(double v) { return static_cast<data_t>(v); }
static weight_t wfrom(double v) { return static_cast<weight_t>(v); }

static double max_abs_diff(const std::vector<data_t> &a, const FloatTensor &b) {
    double m = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i)
        m = std::max(m, std::abs((double)a[i] - b[i]));
    return m;
}

#endif

// ---------------------------------------------------------------------------
// Argmax unit cases (shared by both build modes). Values are multiples of
// 1/32 so they convert to fixed raw exactly.
// ---------------------------------------------------------------------------

static int check_argmax_cases() {
    struct Case { const char *name; double v[NUM_CLASSES]; int expect; };
    const Case cases[] = {
        {"unique max @3",  {0, .25, -.5, 1.25, .5, -.25, .75, -.75, .125, -.125}, 3},
        {"tie 2&5 -> 2",   {0, .25, 1.0, .5, -.5, 1.0, .75, -.75, .125, -.125},  2},
        {"all equal -> 0", {.5, .5, .5, .5, .5, .5, .5, .5, .5, .5},              0},
        {"max @9",         {0, .25, -.5, .5, -.25, .75, -.75, .125, -.125, 1.5}, 9},
        {"all negative",   {-2, -1.5, -3, -1.25, -2.5, -1.75, -3.5, -.875, -2.25, -1.125}, 7},
    };
    int bad = 0;
    for (const Case &c : cases) {
        data_t logits[NUM_CLASSES];
#ifdef LENET_USE_FIXED
        for (int i = 0; i < NUM_CLASSES; ++i)
            logits[i] = dfrom((Raw)std::lrint(c.v[i] * 32.0)); // 1/32 grid, exact
#else
        for (int i = 0; i < NUM_CLASSES; ++i) logits[i] = dfrom(c.v[i]);
#endif
        const int got = argmax10(logits);
        if (got != c.expect) {
            std::cout << "  argmax case '" << c.name << "' got " << got
                      << " expect " << c.expect << "\n";
            ++bad;
        }
    }
    std::cout << "  argmax unit cases: " << (5 - bad) << "/5 pass\n";
    return bad;
}

// ---------------------------------------------------------------------------

#ifndef LENET_USE_FIXED

static void check_float() {
    // 1) relu1 + pool1 against member2 float reference (exact ops).
    std::cout << "[Float] relu1+pool1 vs member2 reference (5 samples)...\n";
    double err1 = 0.0;
    for (int s = 0; s < 5; ++s) {
        const std::string dir = sample_dir(M5_REF_FLOAT_DIR, s);
        FloatTensor conv1 = load_float(dir + "conv1.txt", CONV1_OUT_SIZE);
        FloatTensor pool1 = load_float(dir + "pool1.txt", POOL1_OUT_SIZE);

        std::vector<data_t> d1(CONV1_OUT_SIZE), r1(CONV1_OUT_SIZE), p1(POOL1_OUT_SIZE);
        for (std::size_t i = 0; i < conv1.size(); ++i) d1[i] = dfrom(conv1[i]);
        relu<CONV1_OUT_SIZE>(d1.data(), r1.data());
        maxpool2d<POOL1_C, POOL1_IN_H, POOL1_IN_W>(r1.data(), p1.data());

        err1 = std::max(err1, max_abs_diff(p1, pool1));
    }
    std::cout << "[Float] relu1+pool1 max abs err = " << err1 << "\n";
    if (err1 > 1e-6) throw std::runtime_error("pool1 float error too large");

    // 2) relu2 + pool2 + flatten against member5 data references; keep the
    //    flattened output as the FC chain input.
    std::cout << "[Float] relu2+pool2+flatten vs member5 references (5 samples)...\n";
    double err2 = 0.0;
    std::vector<std::vector<data_t>> flat(5);
    for (int s = 0; s < 5; ++s) {
        const std::string dir = sample_dir(M5_REF_FLOAT_DIR, s);
        const std::string ref = sample_dir(M5_DATA_DIR "reference/float", s);
        FloatTensor conv2 = load_float(dir + "conv2.txt", CONV2_OUT_SIZE);
        FloatTensor r2ref = load_float(ref + "relu2.txt", CONV2_OUT_SIZE);
        FloatTensor p2ref = load_float(ref + "pool2.txt", POOL2_OUT_SIZE);
        FloatTensor f2ref = load_float(ref + "flatten.txt", FLATTEN_SIZE);

        std::vector<data_t> d2(CONV2_OUT_SIZE), r2(CONV2_OUT_SIZE), p2(POOL2_OUT_SIZE), f2(FLATTEN_SIZE);
        for (std::size_t i = 0; i < conv2.size(); ++i) d2[i] = dfrom(conv2[i]);
        relu<CONV2_OUT_SIZE>(d2.data(), r2.data());
        maxpool2d<POOL2_C, POOL2_IN_H, POOL2_IN_W>(r2.data(), p2.data());
        flatten<FLATTEN_C, FLATTEN_H, FLATTEN_W>(p2.data(), f2.data());

        err2 = std::max(err2, max_abs_diff(r2, r2ref));
        err2 = std::max(err2, max_abs_diff(p2, p2ref));
        err2 = std::max(err2, max_abs_diff(f2, f2ref));
        flat[s] = f2;
    }
    std::cout << "[Float] relu2+pool2+flatten max abs err = " << err2 << "\n";
    if (err2 > 1e-6) throw std::runtime_error("pool2/flatten float error too large");

    // 3) FC chain with the real weights: fc1 -> relu3 -> fc2 -> relu4 ->
    //    logits -> argmax, stage-by-stage vs member5 references.
    std::cout << "[Float] fc chain (real weights) vs member5 references (5 samples)...\n";
    FloatTensor fw1 = load_float(std::string(M5_DATA_DIR) + "weights/float/fc1.weight.txt", FC1_WEIGHT_COUNT);
    FloatTensor fw2 = load_float(std::string(M5_DATA_DIR) + "weights/float/fc2.weight.txt", FC2_WEIGHT_COUNT);
    FloatTensor fw3 = load_float(std::string(M5_DATA_DIR) + "weights/float/fc3.weight.txt", FC3_WEIGHT_COUNT);
    std::vector<weight_t> w1(FC1_WEIGHT_COUNT), w2(FC2_WEIGHT_COUNT), w3(FC3_WEIGHT_COUNT);
    for (int i = 0; i < FC1_WEIGHT_COUNT; ++i) w1[i] = wfrom(fw1[i]);
    for (int i = 0; i < FC2_WEIGHT_COUNT; ++i) w2[i] = wfrom(fw2[i]);
    for (int i = 0; i < FC3_WEIGHT_COUNT; ++i) w3[i] = wfrom(fw3[i]);

    double err_stage[5] = {0, 0, 0, 0, 0};
    for (int s = 0; s < 5; ++s) {
        const std::string ref = sample_dir(M5_DATA_DIR "reference/float", s);
        FloatTensor fc1ref = load_float(ref + "fc1.txt", FC1_OUT);
        FloatTensor r3ref = load_float(ref + "relu3.txt", FC1_OUT);
        FloatTensor fc2ref = load_float(ref + "fc2.txt", FC2_OUT);
        FloatTensor r4ref = load_float(ref + "relu4.txt", FC2_OUT);
        FloatTensor lgref = load_float(ref + "logits.txt", FC3_OUT);

        std::vector<data_t> y1(FC1_OUT), y1t(FC1_OUT), y2(FC2_OUT), y3(FC3_OUT);
        fc<FC1_OUT, FC1_IN>(flat[s].data(), w1.data(), y1.data());
        fc_tiled<FC1_OUT, FC1_IN, 8>(flat[s].data(), (const weight_t(*)[8 * FC1_IN])w1.data(), y1t.data());
        for (int i = 0; i < FC1_OUT; ++i)
            if ((double)y1t[i] != (double)y1[i])
                throw std::runtime_error("fc_tiled != fc at fc1 out " + std::to_string(i));
        err_stage[0] = std::max(err_stage[0], max_abs_diff(y1, fc1ref));
        relu<FC1_OUT>(y1.data(), y1.data());
        err_stage[1] = std::max(err_stage[1], max_abs_diff(y1, r3ref));
        fc<FC2_OUT, FC2_IN>(y1.data(), w2.data(), y2.data());
        err_stage[2] = std::max(err_stage[2], max_abs_diff(y2, fc2ref));
        relu<FC2_OUT>(y2.data(), y2.data());
        err_stage[3] = std::max(err_stage[3], max_abs_diff(y2, r4ref));
        fc<FC3_OUT, FC3_IN>(y2.data(), w3.data(), y3.data());
        err_stage[4] = std::max(err_stage[4], max_abs_diff(y3, lgref));

        int g_arg = 0;
        for (int i = 1; i < FC3_OUT; ++i) if (lgref[i] > lgref[g_arg]) g_arg = i;
        std::cout << "  sample " << s << ": argmax hls=" << argmax10(y3.data())
                  << " ref=" << g_arg << "\n";
        if (argmax10(y3.data()) != g_arg)
            throw std::runtime_error("argmax mismatch in sample " + std::to_string(s));
    }
    std::cout << "[Float] stage max abs err fc1=" << err_stage[0]
              << " relu3=" << err_stage[1] << " fc2=" << err_stage[2]
              << " relu4=" << err_stage[3] << " logits=" << err_stage[4] << "\n";
    for (double e : err_stage)
        if (e > 1e-3) throw std::runtime_error("fc float error too large");

    // 4) argmax unit cases.
    if (check_argmax_cases()) throw std::runtime_error("argmax unit cases failed");
}

#else // LENET_USE_FIXED

static void check_fixed() {
    // 1) relu1 + pool1 against member3 fixed raw reference (0 mismatch).
    std::cout << "[Fixed] relu1+pool1 vs member3 raw reference (5 samples)...\n";
    for (int s = 0; s < 5; ++s) {
        const std::string dir = sample_dir(M5_REF_FIXED_DIR, s);
        RawTensor conv1 = load_raw(dir + "conv1.txt", CONV1_OUT_SIZE);
        RawTensor pool1 = load_raw(dir + "pool1.txt", POOL1_OUT_SIZE);

        std::vector<data_t> d1(CONV1_OUT_SIZE), r1(CONV1_OUT_SIZE), p1(POOL1_OUT_SIZE);
        for (std::size_t i = 0; i < conv1.size(); ++i) d1[i] = dfrom(conv1[i]);
        relu<CONV1_OUT_SIZE>(d1.data(), r1.data());
        maxpool2d<POOL1_C, POOL1_IN_H, POOL1_IN_W>(r1.data(), p1.data());

        const std::size_t mism = raw_mismatch(p1, pool1);
        std::cout << "  sample " << s << ": pool1 raw mismatch = " << mism
                  << " / " << pool1.size() << "\n";
        if (mism) throw std::runtime_error("pool1 raw mismatch in sample " + std::to_string(s));
    }

    // 2) relu2 + pool2 + flatten against member5 raw references (0 mismatch).
    std::cout << "[Fixed] relu2+pool2+flatten vs member5 raw references (5 samples)...\n";
    std::vector<std::vector<data_t>> flat(5);
    for (int s = 0; s < 5; ++s) {
        const std::string dir = sample_dir(M5_REF_FIXED_DIR, s);
        const std::string ref = sample_dir(M5_DATA_DIR "reference/fixed", s);
        RawTensor conv2 = load_raw(dir + "conv2.txt", CONV2_OUT_SIZE);
        RawTensor r2ref = load_raw(ref + "relu2.txt", CONV2_OUT_SIZE);
        RawTensor p2ref = load_raw(ref + "pool2.txt", POOL2_OUT_SIZE);
        RawTensor f2ref = load_raw(ref + "flatten.txt", FLATTEN_SIZE);

        std::vector<data_t> d2(CONV2_OUT_SIZE), r2(CONV2_OUT_SIZE), p2(POOL2_OUT_SIZE), f2(FLATTEN_SIZE);
        for (std::size_t i = 0; i < conv2.size(); ++i) d2[i] = dfrom(conv2[i]);
        relu<CONV2_OUT_SIZE>(d2.data(), r2.data());
        maxpool2d<POOL2_C, POOL2_IN_H, POOL2_IN_W>(r2.data(), p2.data());
        flatten<FLATTEN_C, FLATTEN_H, FLATTEN_W>(p2.data(), f2.data());

        const std::size_t mism = raw_mismatch(r2, r2ref) + raw_mismatch(p2, p2ref) + raw_mismatch(f2, f2ref);
        std::cout << "  sample " << s << ": relu2/pool2/flatten raw mismatch = " << mism
                  << " / " << (CONV2_OUT_SIZE + POOL2_OUT_SIZE + FLATTEN_SIZE) << "\n";
        if (mism) throw std::runtime_error("pool2/flatten raw mismatch in sample " + std::to_string(s));
        flat[s] = f2;
    }

    // 3) FC chain with the real quantized weights (0 mismatch, member3
    //    narrowing is built into gemm_systolic).
    std::cout << "[Fixed] fc chain (real raw weights) vs member5 raw references (5 samples)...\n";
    RawTensor rw1 = load_raw(std::string(M5_DATA_DIR) + "weights/fixed/fc1.weight.txt", FC1_WEIGHT_COUNT);
    RawTensor rw2 = load_raw(std::string(M5_DATA_DIR) + "weights/fixed/fc2.weight.txt", FC2_WEIGHT_COUNT);
    RawTensor rw3 = load_raw(std::string(M5_DATA_DIR) + "weights/fixed/fc3.weight.txt", FC3_WEIGHT_COUNT);
    std::vector<weight_t> w1(FC1_WEIGHT_COUNT), w2(FC2_WEIGHT_COUNT), w3(FC3_WEIGHT_COUNT);
    for (int i = 0; i < FC1_WEIGHT_COUNT; ++i) w1[i] = wfrom(rw1[i]);
    for (int i = 0; i < FC2_WEIGHT_COUNT; ++i) w2[i] = wfrom(rw2[i]);
    for (int i = 0; i < FC3_WEIGHT_COUNT; ++i) w3[i] = wfrom(rw3[i]);

    for (int s = 0; s < 5; ++s) {
        const std::string ref = sample_dir(M5_DATA_DIR "reference/fixed", s);
        RawTensor fc1ref = load_raw(ref + "fc1.txt", FC1_OUT);
        RawTensor r3ref = load_raw(ref + "relu3.txt", FC1_OUT);
        RawTensor fc2ref = load_raw(ref + "fc2.txt", FC2_OUT);
        RawTensor r4ref = load_raw(ref + "relu4.txt", FC2_OUT);
        RawTensor lgref = load_raw(ref + "logits.txt", FC3_OUT);

        std::vector<data_t> y1(FC1_OUT), y1t(FC1_OUT), y2(FC2_OUT), y3(FC3_OUT);
        fc<FC1_OUT, FC1_IN>(flat[s].data(), w1.data(), y1.data());
        fc_tiled<FC1_OUT, FC1_IN, 8>(flat[s].data(), (const weight_t(*)[8 * FC1_IN])w1.data(), y1t.data());
        std::size_t mism = raw_mismatch(y1t, fc1ref);
        for (int i = 0; i < FC1_OUT; ++i)
            if (rfrom(y1t[i]) != rfrom(y1[i])) ++mism;
        relu<FC1_OUT>(y1.data(), y1.data());
        mism += raw_mismatch(y1, r3ref);
        fc<FC2_OUT, FC2_IN>(y1.data(), w2.data(), y2.data());
        mism += raw_mismatch(y2, fc2ref);
        relu<FC2_OUT>(y2.data(), y2.data());
        mism += raw_mismatch(y2, r4ref);
        fc<FC3_OUT, FC3_IN>(y2.data(), w3.data(), y3.data());
        mism += raw_mismatch(y3, lgref);

        const std::size_t total = 3 * FC1_OUT + 2 * FC2_OUT + FC3_OUT;
        std::cout << "  sample " << s << ": fc chain raw mismatch = " << mism
                  << " / " << total << "\n";
        if (mism) throw std::runtime_error("fc chain raw mismatch in sample " + std::to_string(s));

        int g_arg = 0;
        for (int i = 1; i < FC3_OUT; ++i) if (lgref[i] > lgref[g_arg]) g_arg = i;
        std::cout << "  sample " << s << ": argmax hls=" << argmax10(y3.data())
                  << " ref=" << g_arg << "\n";
        if (argmax10(y3.data()) != g_arg)
            throw std::runtime_error("argmax mismatch in sample " + std::to_string(s));
    }

    // 4) argmax unit cases.
    if (check_argmax_cases()) throw std::runtime_error("argmax unit cases failed");
}

#endif

int main() {
    try {
#ifdef LENET_USE_FIXED
        check_fixed();
        std::cout << "MEMBER5 FIXED PASS\n";
#else
        check_float();
        std::cout << "MEMBER5 FLOAT PASS\n";
#endif
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
