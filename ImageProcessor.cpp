#include "ImageProcessor.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize2.h"

// Constants for DCT
const int RESIZE_WIDTH = 32;
const int RESIZE_HEIGHT = 32;
const int DCT_SIZE = 8;
const double PI = 3.14159265358979323846;

std::vector<float> ImageProcessor::loadAndPreprocess(const std::string& filepath) {
    int width, height, channels;
    // Load as grayscale (1 channel)
    unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
    if (!img) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    std::vector<unsigned char> resized_img(RESIZE_WIDTH * RESIZE_HEIGHT);
    
    // Resize image to 32x32 using stb_image_resize2
    stbir_resize_uint8_linear(img, width, height, 0,
                              resized_img.data(), RESIZE_WIDTH, RESIZE_HEIGHT, 0,
                              (stbir_pixel_layout)1); // STBIR_1CHANNEL = 1 in enum

    stbi_image_free(img);

    std::vector<float> float_pixels(RESIZE_WIDTH * RESIZE_HEIGHT);
    for (size_t i = 0; i < resized_img.size(); ++i) {
        float_pixels[i] = static_cast<float>(resized_img[i]);
    }

    return float_pixels;
}

uint64_t ImageProcessor::computeHashFromPixels(const std::vector<float>& pixels) {
    std::vector<float> dct_result(DCT_SIZE * DCT_SIZE, 0.0f);

    static bool cosines_initialized = false;
    static std::vector<std::vector<float>> cosines(RESIZE_WIDTH, std::vector<float>(RESIZE_WIDTH));
    if (!cosines_initialized) {
        for (int i = 0; i < RESIZE_WIDTH; ++i) {
            for (int j = 0; j < RESIZE_WIDTH; ++j) {
                cosines[i][j] = static_cast<float>(std::cos((2 * j + 1) * i * PI / (2.0 * RESIZE_WIDTH)));
            }
        }
        cosines_initialized = true;
    }

    // 2D DCT for top-left 8x8
    for (int u = 0; u < DCT_SIZE; ++u) {
        for (int v = 0; v < DCT_SIZE; ++v) {
            float sum = 0.0f;
            for (int i = 0; i < RESIZE_WIDTH; ++i) {
                for (int j = 0; j < RESIZE_WIDTH; ++j) {
                    sum += pixels[i * RESIZE_WIDTH + j] * cosines[u][i] * cosines[v][j];
                }
            }
            
            float cu = (u == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            float cv = (v == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            
            dct_result[u * DCT_SIZE + v] = 0.25f * cu * cv * sum;
        }
    }

    // Calculate median, excluding the DC component at (0,0)
    std::vector<float> dct_flat;
    dct_flat.reserve(DCT_SIZE * DCT_SIZE - 1);
    for (int u = 0; u < DCT_SIZE; ++u) {
        for (int v = 0; v < DCT_SIZE; ++v) {
            if (u == 0 && v == 0) continue;
            dct_flat.push_back(dct_result[u * DCT_SIZE + v]);
        }
    }

    std::nth_element(dct_flat.begin(), dct_flat.begin() + dct_flat.size() / 2, dct_flat.end());
    float median = dct_flat[dct_flat.size() / 2];

    uint64_t hash = 0;
    int bit_index = 0;
    for (int u = 0; u < DCT_SIZE; ++u) {
        for (int v = 0; v < DCT_SIZE; ++v) {
            // Generating 64-bit hash
            if (dct_result[u * DCT_SIZE + v] > median) {
                hash |= (1ULL << bit_index);
            }
            bit_index++;
        }
    }

    return hash;
}

uint64_t ImageProcessor::generatePHash(const std::string& filepath) {
    auto pixels = loadAndPreprocess(filepath);
    return computeHashFromPixels(pixels);
}
