// Host-side batch adapter for the HLS Level 2 preprocessing kernels.
// It decodes no images: input is P6 PPM produced by export_level2_rgb.ps1.
// All grayscale/ROI/resize/centering/quantization calls are in hls/preprocess/.

#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <direct.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../hls/preprocess/level2_preprocess.h"

struct ManifestRow {
    std::string filename;
    std::string label;
    std::string writer_id;
    std::string folder;
    std::string source_relative;
    std::string rgb_relative;
};

static std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::string::size_type i = 0; i < line.size(); ++i) {
        const char character = line[i];
        if (character == '"') {
            quoted = !quoted;
        } else if (character == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field += character;
        }
    }
    fields.push_back(field);
    return fields;
}

static std::string join_path(const std::string &left, const std::string &right) {
    if (left.empty()) return right;
    const char last = left[left.size() - 1];
    if (last == '/' || last == '\\') return left + right;
    return left + "/" + right;
}

static std::string replace_extension(const std::string &filename, const std::string &extension) {
    const std::string::size_type dot = filename.find_last_of('.');
    return (dot == std::string::npos ? filename : filename.substr(0, dot)) + extension;
}

static void create_directory_chain(const std::string &path) {
    std::string current;
    for (std::string::size_type i = 0; i < path.size(); ++i) {
        const char character = path[i];
        current += character;
        if ((character == '/' || character == '\\') && current.size() > 1 &&
            !(current.size() == 3 && current[1] == ':')) {
            if (_mkdir(current.c_str()) != 0 && errno != EEXIST)
                throw std::runtime_error("cannot create directory: " + current);
        }
    }
    if (!current.empty() && _mkdir(current.c_str()) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create directory: " + current);
}

static void ensure_parent_directory(const std::string &filename) {
    const std::string::size_type separator = filename.find_last_of("/\\");
    if (separator != std::string::npos) create_directory_chain(filename.substr(0, separator));
}

static std::vector<ManifestRow> read_manifest(const std::string &manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) throw std::runtime_error("cannot open manifest: " + manifest_path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty manifest");
    std::vector<ManifestRow> rows;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> values = split_csv(line);
        if (values.size() != 8) throw std::runtime_error("invalid manifest row: " + line);
        rows.push_back({values[0], values[1], values[2], values[3], values[4], values[5]});
    }
    return rows;
}

static void read_ppm_rgb(const std::string &path, std::vector<uint8_t> &rgb, int &width, int &height) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open PPM: " + path);
    std::string magic;
    int maximum = 0;
    input >> magic >> width >> height >> maximum;
    input.get();  // consume the single LF after the P6 header written by the adapter.
    if (!input || magic != "P6" || maximum != 255 || width <= 0 || height <= 0 ||
        width > LEVEL2_MAX_WIDTH || height > LEVEL2_MAX_HEIGHT) {
        throw std::runtime_error("unsupported PPM or dimensions: " + path);
    }
    rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3U);
    input.read(reinterpret_cast<char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (input.gcount() != static_cast<std::streamsize>(rgb.size()))
        throw std::runtime_error("truncated PPM: " + path);
}

static void write_pgm(const std::string &path, const uint8_t image[LEVEL2_OUTPUT_PIXELS]) {
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write PGM: " + path);
    output << "P5\n28 28\n255\n";
    output.write(reinterpret_cast<const char *>(image), LEVEL2_OUTPUT_PIXELS);
}

static void write_raw_text(const std::string &path, const int16_t raw[LEVEL2_OUTPUT_PIXELS]) {
    ensure_parent_directory(path);
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write fixed input: " + path);
    for (int i = 0; i < LEVEL2_OUTPUT_PIXELS; ++i) output << raw[i] << '\n';
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: level2_preprocess_driver <raw_rgb_root> <manifest.csv> <self_collected_root>\n";
        return 2;
    }
    try {
        const std::string raw_root(argv[1]);
        const std::string manifest_path(argv[2]);
        const std::string output_root(argv[3]);
        const std::vector<ManifestRow> rows = read_manifest(manifest_path);
        const std::string output_manifest = join_path(output_root, "level2_manifest.csv");
        std::ofstream manifest(output_manifest);
        if (!manifest) throw std::runtime_error("cannot write output manifest");
        manifest << "filename,label,writer_id,folder,source_relative,simple_relative,full_relative,"
                    "fixed_simple_relative,fixed_full_relative,full_roi_valid,full_foreground_pixels,"
                    "full_threshold,full_roi_left,full_roi_top,full_roi_right,full_roi_bottom\n";

        for (const ManifestRow &row : rows) {
            std::vector<uint8_t> rgb;
            int width = 0;
            int height = 0;
            read_ppm_rgb(join_path(raw_root, row.rgb_relative), rgb, width, height);
            uint8_t simple[LEVEL2_OUTPUT_PIXELS];
            uint8_t full[LEVEL2_OUTPUT_PIXELS];
            int16_t simple_raw[LEVEL2_OUTPUT_PIXELS];
            int16_t full_raw[LEVEL2_OUTPUT_PIXELS];
            Level2FullPreprocessInfo info;
            if (level2_simple_preprocess_rgb(rgb.data(), width, height, simple) != 0 ||
                level2_full_preprocess_rgb(rgb.data(), width, height, full, info) != 0) {
                throw std::runtime_error("HLS preprocessing failed: " + row.filename);
            }
            level2_export_a12_raw(simple, simple_raw);
            level2_export_a12_raw(full, full_raw);

            const std::string simple_relative = join_path(join_path("processed_simple", row.folder), replace_extension(row.filename, ".pgm"));
            const std::string full_relative = join_path(join_path("processed_full", row.folder), replace_extension(row.filename, ".pgm"));
            const std::string fixed_simple_relative = join_path(join_path(join_path("fixed_input", "simple"), row.folder), replace_extension(row.filename, ".txt"));
            const std::string fixed_full_relative = join_path(join_path(join_path("fixed_input", "full"), row.folder), replace_extension(row.filename, ".txt"));
            const std::string simple_path = join_path(output_root, simple_relative);
            const std::string full_path = join_path(output_root, full_relative);
            const std::string fixed_simple_path = join_path(output_root, fixed_simple_relative);
            const std::string fixed_full_path = join_path(output_root, fixed_full_relative);
            write_pgm(simple_path, simple);
            write_pgm(full_path, full);
            write_raw_text(fixed_simple_path, simple_raw);
            write_raw_text(fixed_full_path, full_raw);

            manifest << row.filename << ',' << row.label << ',' << row.writer_id << ',' << row.folder << ','
                     << row.source_relative << ','
                     << simple_relative << ',' << full_relative << ','
                     << fixed_simple_relative << ',' << fixed_full_relative << ','
                     << info.valid_roi << ',' << info.foreground_pixels << ',' << info.threshold << ','
                     << info.roi_left << ',' << info.roi_top << ',' << info.roi_right << ',' << info.roi_bottom << '\n';
            std::cout << "processed " << row.folder << '/' << row.filename << "\n";
        }
        std::cout << "PASS: " << rows.size() << " Level 2 samples; manifest=" << output_manifest << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
