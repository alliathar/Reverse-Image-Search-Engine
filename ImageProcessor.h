#pragma once

#include <string>
#include <cstdint>
#include <vector>

class ImageProcessor {
public:
    // --- CNN embedding (MobileNetV3-Small via ONNX Runtime) ---
    // Lazily loads the ONNX model on first call.
    static void initialize(const std::string& modelPath);
    // 576-d L2-normalized feature vector. Distance is cosine.
    static std::vector<float> generateEmbedding(const std::string& filepath);
    // Number of dimensions in the embedding. Valid after initialize().
    static size_t embeddingDim();

    // --- 64-bit DCT pHash (legacy) ---
    // Produces a 64-bit perceptual hash. Distance is Hamming popcount.
    static uint64_t generatePHash(const std::string& filepath);
};
