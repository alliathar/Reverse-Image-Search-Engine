#pragma once

#include <string>
#include <cstdint>
#include <vector>

class ImageProcessor {
public:
    // Creates a 64-bit pHash from a given image filepath using full 2D DCT
    static uint64_t generatePHash(const std::string& filepath);

private:
    // Load via stb_image and resize to 32x32, returning grayscale float map
    static std::vector<float> loadAndPreprocess(const std::string& filepath);

    // 2D Discrete Cosine Transform on 32x32 image and extract top-left 8x8 to produce 64-bit hash
    static uint64_t computeHashFromPixels(const std::vector<float>& pixels);
};
