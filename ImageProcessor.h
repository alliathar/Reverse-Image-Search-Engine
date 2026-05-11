#pragma once

#include <string>
#include <vector>

class ImageProcessor {
public:
    // Lazily loads the ONNX model on first call. Must be called from the same process
    // that runs inference. Safe to call multiple times.
    static void initialize(const std::string& modelPath);

    // Loads the image, preprocesses (resize 224x224, RGB, ImageNet normalize),
    // runs inference, and returns the L2-normalized embedding (576-d for MobileNetV3-Small).
    static std::vector<float> generateEmbedding(const std::string& filepath);

    // Number of dimensions in the embedding. Valid after initialize().
    static size_t embeddingDim();
};
