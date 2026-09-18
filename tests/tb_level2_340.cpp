// Level 2 self-collected fixed-input driver for the optimized LeNet top.
//
// This testbench reads data/self_collected/level2_manifest.csv and consumes the
// fixed raw input files named by fixed_simple_relative/fixed_full_relative.
// It intentionally does not preprocess images or requantize PGM pixels.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../hls/operators/argmax10.h"
#include "../hls/top/lenet_accelerator.h"

#ifndef LENET_USE_FIXED
#error "tb_level2_340.cpp must be compiled with LENET_USE_FIXED"
#endif

#include <ap_int.h>

using Raw = std::int64_t;

struct Options {
    std::string manifest;
    std::string data_root;
    std::string weights;
    std::string output;
    std::string mode;
};

struct Sample {
    std::string sample_id;
    int label;
    std::string fixed_relative;
};

static void usage(const char *program) {
    std::cerr << "Usage: " << program
              << " --manifest FILE --data-root DIR --weights DIR"
                 " --mode simple|full --output FILE\n";
}

static Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--manifest" || key == "--data-root" || key == "--weights" ||
            key == "--output" || key == "--mode") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            const std::string value(argv[++i]);
            if (key == "--manifest") options.manifest = value;
            else if (key == "--data-root") options.data_root = value;
            else if (key == "--weights") options.weights = value;
            else if (key == "--output") options.output = value;
            else options.mode = value;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    if (options.manifest.empty() || options.data_root.empty() || options.weights.empty() ||
        options.output.empty() || (options.mode != "simple" && options.mode != "full")) {
        usage(argv[0]);
        throw std::runtime_error("missing required options");
    }
    return options;
}

static std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!line.empty() && line[line.size() - 1] == ',') fields.push_back("");
    return fields;
}

static std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    const char last = a[a.size() - 1];
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

template <typename T>
static std::vector<T> load_text(const std::string &path, std::size_t expected) {
    std::ifstream stream(path.c_str());
    if (!stream) throw std::runtime_error("cannot open " + path);
    std::vector<T> values;
    T value;
    while (stream >> value) values.push_back(value);
    if (values.size() != expected) {
        std::ostringstream message;
        message << path << ": expected " << expected << " values, got " << values.size();
        throw std::runtime_error(message.str());
    }
    return values;
}

static std::vector<Sample> read_manifest(const Options &options) {
    std::ifstream stream(options.manifest.c_str());
    if (!stream) throw std::runtime_error("cannot open manifest " + options.manifest);
    std::string line;
    if (!std::getline(stream, line)) throw std::runtime_error("empty manifest");
    const std::vector<std::string> header = split_csv_line(line);
    std::map<std::string, std::size_t> column;
    for (std::size_t i = 0; i < header.size(); ++i) column[header[i]] = i;
    const std::string fixed_column = options.mode == "simple" ?
        "fixed_simple_relative" : "fixed_full_relative";
    for (const char *name : {"sample_id", "label"}) {
        if (!column.count(name)) throw std::runtime_error(std::string("missing manifest column ") + name);
    }
    if (!column.count(fixed_column)) throw std::runtime_error("missing manifest column " + fixed_column);

    std::vector<Sample> samples;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() < header.size()) throw std::runtime_error("short manifest row");
        Sample sample;
        sample.sample_id = fields[column["sample_id"]];
        sample.label = std::atoi(fields[column["label"]].c_str());
        sample.fixed_relative = fields[column[fixed_column]];
        samples.push_back(sample);
    }
    return samples;
}

static std::string weight_path(const std::string &root, const char *name) {
    return join_path(root, std::string(name) + ".weight.txt");
}

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

static void run(const Options &options) {
    const std::vector<Sample> samples = read_manifest(options);
    if (samples.size() != 340) {
        std::ostringstream message;
        message << "expected 340 samples, got " << samples.size();
        throw std::runtime_error(message.str());
    }

    const std::vector<Raw> conv1_values = load_text<Raw>(weight_path(options.weights, "conv1"), CONV1_WEIGHT_COUNT);
    const std::vector<Raw> conv2_values = load_text<Raw>(weight_path(options.weights, "conv2"), CONV2_WEIGHT_COUNT);
    const std::vector<Raw> fc1_values = load_text<Raw>(weight_path(options.weights, "fc1"), FC1_WEIGHT_COUNT);
    const std::vector<Raw> fc2_values = load_text<Raw>(weight_path(options.weights, "fc2"), FC2_WEIGHT_COUNT);
    const std::vector<Raw> fc3_values = load_text<Raw>(weight_path(options.weights, "fc3"), FC3_WEIGHT_COUNT);

    std::vector<weight_t> conv1_w(CONV1_WEIGHT_COUNT), conv2_w(CONV2_WEIGHT_COUNT);
    std::vector<weight_t> fc1_storage(FC1_WEIGHT_COUNT), fc2_storage(FC2_WEIGHT_COUNT), fc3_storage(FC3_WEIGHT_COUNT);
    for (int i = 0; i < CONV1_WEIGHT_COUNT; ++i) conv1_w[i] = weight_from_raw(conv1_values[i]);
    for (int i = 0; i < CONV2_WEIGHT_COUNT; ++i) conv2_w[i] = weight_from_raw(conv2_values[i]);
    for (int i = 0; i < FC1_WEIGHT_COUNT; ++i) fc1_storage[i] = weight_from_raw(fc1_values[i]);
    for (int i = 0; i < FC2_WEIGHT_COUNT; ++i) fc2_storage[i] = weight_from_raw(fc2_values[i]);
    for (int i = 0; i < FC3_WEIGHT_COUNT; ++i) fc3_storage[i] = weight_from_raw(fc3_values[i]);

    const weight_t (*fc1_w)[8 * FC1_IN] =
        reinterpret_cast<const weight_t (*)[8 * FC1_IN]>(fc1_storage.data());
    const weight_t (*fc2_w)[12 * FC2_IN] =
        reinterpret_cast<const weight_t (*)[12 * FC2_IN]>(fc2_storage.data());
    const weight_t (*fc3_w)[5 * FC3_IN] =
        reinterpret_cast<const weight_t (*)[5 * FC3_IN]>(fc3_storage.data());

    std::ofstream output(options.output.c_str());
    if (!output) throw std::runtime_error("cannot create " + options.output);
    output << "sample_id,label,pred";
    for (int c = 0; c < NUM_CLASSES; ++c) output << ",raw_logit" << c;
    output << "\n";

    for (std::size_t index = 0; index < samples.size(); ++index) {
        const Sample &sample = samples[index];
        const std::vector<Raw> input_raw = load_text<Raw>(
            join_path(options.data_root, sample.fixed_relative), INPUT_SIZE);
        std::vector<data_t> input(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; ++i) input[i] = data_from_raw(input_raw[i]);

        output_t logits[NUM_CLASSES];
        lenet_accelerator_with_weights(input.data(), conv1_w.data(), conv2_w.data(),
                                       fc1_w, fc2_w, fc3_w, logits);
        const int pred = argmax10(logits);
        output << sample.sample_id << "," << sample.label << "," << pred;
        for (int c = 0; c < NUM_CLASSES; ++c) output << "," << raw_from_data(logits[c]);
        output << "\n";
        if ((index + 1) % 50 == 0 || index + 1 == samples.size())
            std::cout << "optimized hls fixed " << options.mode << " "
                      << (index + 1) << "/" << samples.size() << "\n";
    }
}

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        run(options);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
