// Member 6 minimal complete-network testbench.
//
// This testbench supplies the already-exported Member 2/3 weights explicitly.
// It intentionally does not depend on a generated ROM or an AXI memory model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../hls/top/lenet_accelerator.h"
#include "../../hls/operators/argmax10.h"

#ifndef M6_ROOT
#define M6_ROOT ""
#endif

using Raw = std::int64_t;
using RawTensor = std::vector<Raw>;
using FloatTensor = std::vector<double>;

static bool file_exists(const std::string &path) {
    std::ifstream file(path.c_str());
    return file.good();
}

static bool valid_root(const std::string &root) {
    return file_exists(root + "/config/network_config.h") &&
           file_exists(root + "/reference/float/sample_00000/input.txt");
}

static std::string find_root() {
    const std::string fixed_root = M6_ROOT;
    if (!fixed_root.empty()) {
        if (valid_root(fixed_root)) return fixed_root;
        throw std::runtime_error("Invalid M6_ROOT: " + fixed_root);
    }

    const char *env_root = std::getenv("M6_ROOT");
    if (env_root != 0 && env_root[0] != '\0') {
        const std::string root(env_root);
        if (valid_root(root)) return root;
        throw std::runtime_error("Invalid M6_ROOT: " + root);
    }

    static const char *roots[] = {
        ".", "..", "../..", "../../..", "../../../..",
        "../../../../..", "../../../../../..", "../../../../../../..",
        "../../../../../../../.."
    };
    const int root_count = sizeof(roots) / sizeof(roots[0]);
    for (int i = 0; i < root_count; ++i) {
        if (valid_root(roots[i])) return roots[i];
    }

    throw std::runtime_error(
        "Cannot find M6 handoff root; set M6_ROOT to the package path");
}

static std::string sample_dir(const std::string &root, int sample) {
    std::ostringstream path;
    path << root << "/sample_" << std::setw(5) << std::setfill('0')
         << sample << "/";
    return path.str();
}

static RawTensor load_raw(const std::string &path, std::size_t expected) {
    std::ifstream file(path.c_str());
    if (!file) throw std::runtime_error("Cannot open " + path);
    RawTensor values;
    Raw value;
    while (file >> value) values.push_back(value);
    if (values.size() != expected) {
        throw std::runtime_error("Size mismatch " + path + " expected " +
                                 std::to_string(expected) + " got " +
                                 std::to_string(values.size()));
    }
    return values;
}

static FloatTensor load_float(const std::string &path, std::size_t expected) {
    std::ifstream file(path.c_str());
    if (!file) throw std::runtime_error("Cannot open " + path);
    FloatTensor values;
    double value;
    while (file >> value) values.push_back(value);
    if (values.size() != expected) {
        throw std::runtime_error("Size mismatch " + path + " expected " +
                                 std::to_string(expected) + " got " +
                                 std::to_string(values.size()));
    }
    return values;
}

#ifndef LENET_USE_FIXED

static void check_float() {
    const std::string root = find_root();
    const std::string references = root + "/reference/float";
    const std::string conv_weights = root + "/weights/float";
    const std::string fc_data = root;

    const FloatTensor fconv1 = load_float(
        conv_weights + "/conv1.weight.txt", CONV1_WEIGHT_COUNT);
    const FloatTensor fconv2 = load_float(
        conv_weights + "/conv2.weight.txt", CONV2_WEIGHT_COUNT);
    const FloatTensor ffc1 = load_float(
        fc_data + "/weights/float/fc1.weight.txt", FC1_WEIGHT_COUNT);
    const FloatTensor ffc2 = load_float(
        fc_data + "/weights/float/fc2.weight.txt", FC2_WEIGHT_COUNT);
    const FloatTensor ffc3 = load_float(
        fc_data + "/weights/float/fc3.weight.txt", FC3_WEIGHT_COUNT);

    std::vector<weight_t> conv1_w(CONV1_WEIGHT_COUNT);
    std::vector<weight_t> conv2_w(CONV2_WEIGHT_COUNT);
    std::vector<weight_t> fc1_storage(FC1_WEIGHT_COUNT);
    std::vector<weight_t> fc2_storage(FC2_WEIGHT_COUNT);
    std::vector<weight_t> fc3_storage(FC3_WEIGHT_COUNT);
    for (int i = 0; i < CONV1_WEIGHT_COUNT; ++i) conv1_w[i] = fconv1[i];
    for (int i = 0; i < CONV2_WEIGHT_COUNT; ++i) conv2_w[i] = fconv2[i];
    for (int i = 0; i < FC1_WEIGHT_COUNT; ++i) fc1_storage[i] = ffc1[i];
    for (int i = 0; i < FC2_WEIGHT_COUNT; ++i) fc2_storage[i] = ffc2[i];
    for (int i = 0; i < FC3_WEIGHT_COUNT; ++i) fc3_storage[i] = ffc3[i];

    // These views have exactly the same row-major memory order as [OUT][IN].
    const weight_t (*fc1_w)[8 * FC1_IN] =
        (const weight_t (*)[8 * FC1_IN])fc1_storage.data();
    const weight_t (*fc2_w)[12 * FC2_IN] =
        (const weight_t (*)[12 * FC2_IN])fc2_storage.data();
    const weight_t (*fc3_w)[5 * FC3_IN] =
        (const weight_t (*)[5 * FC3_IN])fc3_storage.data();

    double max_error = 0.0;
    for (int sample = 0; sample < 5; ++sample) {
        const std::string input_path = sample_dir(references, sample) +
                                       "input.txt";
        const std::string expected_path = sample_dir(
        fc_data + "/reference/float", sample) + "logits.txt";
        const FloatTensor finput = load_float(input_path, INPUT_SIZE);
        const FloatTensor expected = load_float(expected_path, NUM_CLASSES);

        std::vector<data_t> input(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; ++i) input[i] = finput[i];
        data_t logits[NUM_CLASSES];

        lenet_accelerator_with_weights(
            input.data(), conv1_w.data(), conv2_w.data(), fc1_w, fc2_w,
            fc3_w, logits);

        double sample_error = 0.0;
        int expected_class = 0;
        for (int i = 0; i < NUM_CLASSES; ++i) {
            sample_error = std::max(
                sample_error, std::abs((double)logits[i] - expected[i]));
            if (expected[i] > expected[expected_class]) expected_class = i;
        }
        const int got_class = argmax10(logits);
        max_error = std::max(max_error, sample_error);
        std::cout << "  sample " << sample << ": max_err=" << sample_error
                  << " argmax=" << got_class << " ref=" << expected_class
                  << "\n";
        if (got_class != expected_class)
            throw std::runtime_error("Float argmax mismatch at sample " +
                                     std::to_string(sample));
    }

    std::cout << "[Member6 Float] max abs error = " << max_error << "\n";
    if (max_error > 1e-3)
        throw std::runtime_error("Full network Float error too large");
}

#else

static data_t data_from_raw(Raw raw) {
    data_t value;
    value.range(LENET_DATA_W - 1, 0) = ap_int<LENET_DATA_W>(raw);
    return value;
}

static weight_t weight_from_raw(Raw raw) {
    weight_t value;
    value.range(LENET_WEIGHT_W - 1, 0) = ap_int<LENET_WEIGHT_W>(raw);
    return value;
}

static Raw raw_from_data(data_t value) {
    ap_int<LENET_DATA_W> raw;
    raw.range(LENET_DATA_W - 1, 0) = value.range(LENET_DATA_W - 1, 0);
    return raw.to_int64();
}

static void check_fixed() {
    const std::string root = find_root();
    const std::string references = root + "/reference/fixed";
    const std::string conv_weights = root + "/weights/fixed";
    const std::string fc_data = root;

    const RawTensor rconv1 = load_raw(
        conv_weights + "/conv1.weight.txt", CONV1_WEIGHT_COUNT);
    const RawTensor rconv2 = load_raw(
        conv_weights + "/conv2.weight.txt", CONV2_WEIGHT_COUNT);
    const RawTensor rfc1 = load_raw(
        fc_data + "/weights/fixed/fc1.weight.txt", FC1_WEIGHT_COUNT);
    const RawTensor rfc2 = load_raw(
        fc_data + "/weights/fixed/fc2.weight.txt", FC2_WEIGHT_COUNT);
    const RawTensor rfc3 = load_raw(
        fc_data + "/weights/fixed/fc3.weight.txt", FC3_WEIGHT_COUNT);

    std::vector<weight_t> conv1_w(CONV1_WEIGHT_COUNT);
    std::vector<weight_t> conv2_w(CONV2_WEIGHT_COUNT);
    std::vector<weight_t> fc1_storage(FC1_WEIGHT_COUNT);
    std::vector<weight_t> fc2_storage(FC2_WEIGHT_COUNT);
    std::vector<weight_t> fc3_storage(FC3_WEIGHT_COUNT);
    for (int i = 0; i < CONV1_WEIGHT_COUNT; ++i)
        conv1_w[i] = weight_from_raw(rconv1[i]);
    for (int i = 0; i < CONV2_WEIGHT_COUNT; ++i)
        conv2_w[i] = weight_from_raw(rconv2[i]);
    for (int i = 0; i < FC1_WEIGHT_COUNT; ++i)
        fc1_storage[i] = weight_from_raw(rfc1[i]);
    for (int i = 0; i < FC2_WEIGHT_COUNT; ++i)
        fc2_storage[i] = weight_from_raw(rfc2[i]);
    for (int i = 0; i < FC3_WEIGHT_COUNT; ++i)
        fc3_storage[i] = weight_from_raw(rfc3[i]);

    const weight_t (*fc1_w)[8 * FC1_IN] =
        (const weight_t (*)[8 * FC1_IN])fc1_storage.data();
    const weight_t (*fc2_w)[12 * FC2_IN] =
        (const weight_t (*)[12 * FC2_IN])fc2_storage.data();
    const weight_t (*fc3_w)[5 * FC3_IN] =
        (const weight_t (*)[5 * FC3_IN])fc3_storage.data();

    std::size_t total_mismatch = 0;
    for (int sample = 0; sample < 5; ++sample) {
        const std::string input_path = sample_dir(references, sample) +
                                       "input.txt";
        const std::string expected_path = sample_dir(
        fc_data + "/reference/fixed", sample) + "logits.txt";
        const RawTensor rinput = load_raw(input_path, INPUT_SIZE);
        const RawTensor expected = load_raw(expected_path, NUM_CLASSES);

        std::vector<data_t> input(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; ++i)
            input[i] = data_from_raw(rinput[i]);
        data_t logits[NUM_CLASSES];

        lenet_accelerator_with_weights(
            input.data(), conv1_w.data(), conv2_w.data(), fc1_w, fc2_w,
            fc3_w, logits);

        std::size_t mismatches = 0;
        int expected_class = 0;
        for (int i = 0; i < NUM_CLASSES; ++i) {
            if (raw_from_data(logits[i]) != expected[i]) ++mismatches;
            if (expected[i] > expected[expected_class]) expected_class = i;
        }
        const int got_class = argmax10(logits);
        total_mismatch += mismatches;
        std::cout << "  sample " << sample << ": raw_mismatch=" << mismatches
                  << " argmax=" << got_class << " ref=" << expected_class
                  << "\n";
        if (mismatches != 0)
            throw std::runtime_error("Fixed mismatch at sample " +
                                     std::to_string(sample));
        if (got_class != expected_class)
            throw std::runtime_error("Fixed argmax mismatch at sample " +
                                     std::to_string(sample));
    }

    std::cout << "[Member6 Fixed] raw mismatch = " << total_mismatch << "\n";
    if (total_mismatch != 0)
        throw std::runtime_error("Full network Fixed mismatch");
}

#endif

int main() {
    try {
#ifndef LENET_USE_FIXED
        check_float();
        std::cout << "MEMBER6 FLOAT PASS\n";
#else
        check_fixed();
        std::cout << "MEMBER6 FIXED PASS\n";
#endif
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
