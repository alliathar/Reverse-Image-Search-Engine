#include "ImageProcessor.h"

#include <stdexcept>
#include <cmath>
#include <memory>
#include <algorithm>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize2.h"

#ifndef USE_PHASH
#include <onnxruntime_cxx_api.h>
#endif

namespace {

// ----- pHash constants -----
constexpr int PHASH_RESIZE = 32;
constexpr int PHASH_DCT = 8;
constexpr double PI = 3.14159265358979323846;

#ifndef USE_PHASH

// ----- CNN constants -----
constexpr int CNN_INPUT_W = 224;
constexpr int CNN_INPUT_H = 224;
constexpr int CNN_INPUT_C = 3;

// ImageNet preprocessing constants used by torchvision's pretrained models.
constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

// ----- ONNX session state -----
struct OrtState {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ReverseImageSearch"};
    Ort::SessionOptions session_options{};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
    size_t output_dim = 0;
};

std::unique_ptr<OrtState> g_state;

OrtState& state() {
    if (!g_state) {
        throw std::runtime_error("ImageProcessor::initialize() not called");
    }
    return *g_state;
}

// Load image, resize to 224x224 RGB, normalize to NCHW float tensor.
std::vector<float> loadAndPreprocessCNN(const std::string& filepath) {
    int width = 0, height = 0, channels = 0;
    unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, CNN_INPUT_C);
    if (!img) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    std::vector<unsigned char> resized(CNN_INPUT_W * CNN_INPUT_H * CNN_INPUT_C);
    stbir_resize_uint8_linear(img, width, height, 0,
                              resized.data(), CNN_INPUT_W, CNN_INPUT_H, 0,
                              (stbir_pixel_layout)3); // STBIR_RGB
    stbi_image_free(img);

    std::vector<float> tensor(CNN_INPUT_C * CNN_INPUT_H * CNN_INPUT_W);
    const int plane = CNN_INPUT_H * CNN_INPUT_W;
    for (int y = 0; y < CNN_INPUT_H; ++y) {
        for (int x = 0; x < CNN_INPUT_W; ++x) {
            int src = (y * CNN_INPUT_W + x) * CNN_INPUT_C;
            int dst = y * CNN_INPUT_W + x;
            for (int c = 0; c < CNN_INPUT_C; ++c) {
                float v = resized[src + c] / 255.0f;
                tensor[c * plane + dst] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    return tensor;
}

#endif // !USE_PHASH

// Load image, resize to 32x32 grayscale floats, for the DCT pHash pipeline.
std::vector<float> loadAndPreprocessPHash(const std::string& filepath) {
    int width = 0, height = 0, channels = 0;
    unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
    if (!img) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    std::vector<unsigned char> resized(PHASH_RESIZE * PHASH_RESIZE);
    stbir_resize_uint8_linear(img, width, height, 0,
                              resized.data(), PHASH_RESIZE, PHASH_RESIZE, 0,
                              (stbir_pixel_layout)1); // STBIR_1CHANNEL
    stbi_image_free(img);

    std::vector<float> pixels(PHASH_RESIZE * PHASH_RESIZE);
    for (size_t i = 0; i < resized.size(); ++i) {
        pixels[i] = static_cast<float>(resized[i]);
    }
    return pixels;
}

uint64_t computePHashFromPixels(const std::vector<float>& pixels) {
    std::vector<float> dct_result(PHASH_DCT * PHASH_DCT, 0.0f);

    static bool cosines_initialized = false;
    static std::vector<std::vector<float>> cosines(PHASH_RESIZE, std::vector<float>(PHASH_RESIZE));
    if (!cosines_initialized) {
        for (int i = 0; i < PHASH_RESIZE; ++i) {
            for (int j = 0; j < PHASH_RESIZE; ++j) {
                cosines[i][j] = static_cast<float>(std::cos((2 * j + 1) * i * PI / (2.0 * PHASH_RESIZE)));
            }
        }
        cosines_initialized = true;
    }

    for (int u = 0; u < PHASH_DCT; ++u) {
        for (int v = 0; v < PHASH_DCT; ++v) {
            float sum = 0.0f;
            for (int i = 0; i < PHASH_RESIZE; ++i) {
                for (int j = 0; j < PHASH_RESIZE; ++j) {
                    sum += pixels[i * PHASH_RESIZE + j] * cosines[u][i] * cosines[v][j];
                }
            }
            float cu = (u == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            float cv = (v == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            dct_result[u * PHASH_DCT + v] = 0.25f * cu * cv * sum;
        }
    }

    std::vector<float> dct_flat;
    dct_flat.reserve(PHASH_DCT * PHASH_DCT - 1);
    for (int u = 0; u < PHASH_DCT; ++u) {
        for (int v = 0; v < PHASH_DCT; ++v) {
            if (u == 0 && v == 0) continue;
            dct_flat.push_back(dct_result[u * PHASH_DCT + v]);
        }
    }

    std::nth_element(dct_flat.begin(), dct_flat.begin() + dct_flat.size() / 2, dct_flat.end());
    float median = dct_flat[dct_flat.size() / 2];

    uint64_t hash = 0;
    int bit_index = 0;
    for (int u = 0; u < PHASH_DCT; ++u) {
        for (int v = 0; v < PHASH_DCT; ++v) {
            if (dct_result[u * PHASH_DCT + v] > median) {
                hash |= (1ULL << bit_index);
            }
            bit_index++;
        }
    }
    return hash;
}

} // namespace

#ifndef USE_PHASH

void ImageProcessor::initialize(const std::string& modelPath) {
    if (g_state) return;

    auto s = std::make_unique<OrtState>();
    s->session_options.SetIntraOpNumThreads(1);
    s->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    s->session = std::make_unique<Ort::Session>(s->env, modelPath.c_str(), s->session_options);

    auto input_name_alloc = s->session->GetInputNameAllocated(0, s->allocator);
    auto output_name_alloc = s->session->GetOutputNameAllocated(0, s->allocator);
    s->input_name = input_name_alloc.get();
    s->output_name = output_name_alloc.get();

    Ort::TypeInfo out_type_info = s->session->GetOutputTypeInfo(0);
    auto shape = out_type_info.GetTensorTypeAndShapeInfo().GetShape();
    s->output_dim = static_cast<size_t>(shape.back());

    g_state = std::move(s);
}

size_t ImageProcessor::embeddingDim() {
    return state().output_dim;
}

std::vector<float> ImageProcessor::generateEmbedding(const std::string& filepath) {
    auto& s = state();
    auto input_tensor_values = loadAndPreprocessCNN(filepath);

    std::array<int64_t, 4> input_shape{1, CNN_INPUT_C, CNN_INPUT_H, CNN_INPUT_W};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    const char* input_names[]  = {s.input_name.c_str()};
    const char* output_names[] = {s.output_name.c_str()};

    auto outputs = s.session->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    float* out = outputs[0].GetTensorMutableData<float>();
    std::vector<float> embedding(out, out + s.output_dim);

    // L2-normalize so dot product == cosine similarity.
    double norm_sq = 0.0;
    for (float v : embedding) norm_sq += v * v;
    float norm = static_cast<float>(std::sqrt(norm_sq));
    if (norm > 1e-12f) {
        for (float& v : embedding) v /= norm;
    }
    return embedding;
}

#endif // !USE_PHASH

uint64_t ImageProcessor::generatePHash(const std::string& filepath) {
    auto pixels = loadAndPreprocessPHash(filepath);
    return computePHashFromPixels(pixels);
}
