#include "ImageProcessor.h"

#include <stdexcept>
#include <cmath>
#include <memory>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize2.h"

#include <onnxruntime_cxx_api.h>

namespace {

constexpr int INPUT_W = 224;
constexpr int INPUT_H = 224;
constexpr int INPUT_C = 3;

// ImageNet preprocessing constants used by torchvision's pretrained models.
constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

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

// Load image, resize to 224x224, convert RGB -> NCHW float buffer with ImageNet normalization.
std::vector<float> loadAndPreprocess(const std::string& filepath) {
    int width = 0, height = 0, channels = 0;
    unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, INPUT_C);
    if (!img) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    std::vector<unsigned char> resized(INPUT_W * INPUT_H * INPUT_C);
    stbir_resize_uint8_linear(img, width, height, 0,
                              resized.data(), INPUT_W, INPUT_H, 0,
                              (stbir_pixel_layout)3); // STBIR_RGB
    stbi_image_free(img);

    // Convert HWC uint8 [0,255] -> CHW float normalized.
    std::vector<float> tensor(INPUT_C * INPUT_H * INPUT_W);
    const int plane = INPUT_H * INPUT_W;
    for (int y = 0; y < INPUT_H; ++y) {
        for (int x = 0; x < INPUT_W; ++x) {
            int src = (y * INPUT_W + x) * INPUT_C;
            int dst = y * INPUT_W + x;
            for (int c = 0; c < INPUT_C; ++c) {
                float v = resized[src + c] / 255.0f;
                tensor[c * plane + dst] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    return tensor;
}

} // namespace

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

    // Keep the TypeInfo alive — TensorTypeAndShapeInfo holds a view into it.
    Ort::TypeInfo out_type_info = s->session->GetOutputTypeInfo(0);
    auto shape = out_type_info.GetTensorTypeAndShapeInfo().GetShape();
    // Shape is [batch, dim]; dim is last.
    s->output_dim = static_cast<size_t>(shape.back());

    g_state = std::move(s);
}

size_t ImageProcessor::embeddingDim() {
    return state().output_dim;
}

std::vector<float> ImageProcessor::generateEmbedding(const std::string& filepath) {
    auto& s = state();
    auto input_tensor_values = loadAndPreprocess(filepath);

    std::array<int64_t, 4> input_shape{1, INPUT_C, INPUT_H, INPUT_W};
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
