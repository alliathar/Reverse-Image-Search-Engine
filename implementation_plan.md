# C++ Reverse Image Search Engine (HNSW + pHash)

This plan outlines the architecture for building a fast, modular Reverse Image Search Engine from scratch in Modern C++ (C++17). 

## User Review Required
> [!IMPORTANT]
> Please review the provided boilerplate for `HNSWIndex` and `ImageProcessor` below. The structure establishes a multi-layered proximity graph where each node tracks its neighbors per layer, along with a pHash pipeline leveraging `stb_image`. Once you approve these boilerplate headers, I will move to implement the definitions and unit tests.

## Proposed Architecture

### 1. The HNSW Graph (`HNSW.h` and `HNSW.cpp`)
The Hierarchical Navigable Small World algorithm will use **Hamming distance** for extremely fast similarity comparisons. We will use `__builtin_popcountll` which compiles to an efficient hardware instruction, yielding `O(1)` calculation logic on the 64-bit pHashes.

#### Boilerplate for `HNSW.h`:
```cpp
#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <queue>
#include <unordered_map>
#include <memory>
#include <cmath>

// O(1) Hamming distance for 64-bit integers using compiler intrinsic
inline uint32_t computeHammingDistance(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

struct HNSWNode {
    uint64_t id;
    uint64_t hash; // The 64-bit pHash
    int maxLayer;
    
    // Nearest neighbors for each layer: neighbors[layer][neighbor_index]
    std::vector<std::vector<uint64_t>> neighbors;

    HNSWNode(uint64_t _id, uint64_t _hash, int _maxLayer) 
        : id(_id), hash(_hash), maxLayer(_maxLayer) {
        neighbors.resize(maxLayer + 1);
    }
};

class HNSWIndex {
public:
    // M: max edges per node per level (except bottom layer)
    HNSWIndex(int M = 16, int M_max0 = 32, int efConstruction = 100);
    
    // Inserts an image into the graph
    void insert(uint64_t id, uint64_t hash);
    
    // K-NN Search using a greedy algorithm down through layers
    std::vector<uint64_t> search(uint64_t queryHash, int k, int efSearch = 50);

private:
    int M_;               // Max neighbors per layer (above layer 0)
    int M_max0_;          // Max neighbors at layer 0 (bottom level)
    int efConstruction_;  // Size of the dynamic candidate list during construction
    double levelMult_;    // Multiplier for random layer generation

    uint64_t entryPointId_;
    int maxCurrentLayer_;
    bool hasEntryPoint_;
    
    // Maps unique image IDs to graph nodes
    std::unordered_map<uint64_t, std::shared_ptr<HNSWNode>> nodes_;
    std::mt19937 rng_;

    int generateRandomLayer();
    
    // Standard greedy search for HNSW within a single layer
    std::priority_queue<std::pair<uint32_t, uint64_t>> searchLayer(
        uint64_t queryHash,
        uint64_t entryPoint,
        int ef,
        int layer
    );
};
```

### 2. The Perceptual Hash Pipeline (`ImageProcessor.h` and `ImageProcessor.cpp`)
The `ImageProcessor` handles loading and resizing images using the lightweight header-only properties of `stb_image.h` and `stb_image_resize2.h`. It produces a persistent 64-bit perceptual hash mapping to spatial frequencies.

#### Boilerplate for `ImageProcessor.h`:
```cpp
#pragma once

#include <string>
#include <cstdint>
#include <vector>

class ImageProcessor {
public:
    // Main API: Creates a 64-bit pHash from a given image filepath
    static uint64_t generatePHash(const std::string& filepath);

private:
    // RAII-based load via stb_image and 32x32 resizing
    // Returns a 32x32 grayscale pixel intensity float map
    static std::vector<float> loadAndPreprocess(const std::string& filepath);

    // Simplistic frequency-based hashing (e.g. DCT or Mean-Hash)
    // Converts 32x32 matrix into a representative 64-bit integer
    static uint64_t computeHashFromPixels(const std::vector<float>& pixels);
};
```

## Open Questions

> [!WARNING]
> 1. **DCT vs Mean Hash Design**: True perceptual hashing calculates a highly resilient signature using a 2D Discrete Cosine Transform (taking the low-frequency components). Would you prefer I manually implement the 2D-DCT algorithm on the 32x32 matrix, or stick to a more lightweight block Mean-Hash algorithm for this component?
> 2. **Dependencies**: I plan to use an internet request to fetch `stb_image.h` and `stb_image_resize2.h` directly into a `/third_party` include folder. Does that align with your requirement for "no external libraries"?

## Verification Plan

### Automated Tests
1. **File Hashing Check**: Pass a known image through `ImageProcessor`, log the 64-bit string, slightly modifying the image (e.g., adding noise via command line scripts if needed), and ensuring `HammingDistance < Threshold`.
2. **HNSW Correctness**: Inject 5,000 synthetic random 64-bit "hashes" into `HNSWIndex`. Fire search queries for test hashes and verify against an `O(N)` exhaustive linear search that $K$ nearest neighbors perfectly or near-perfectly match. 
3. Verify graph memory cleanup strictly adheres to C++ RAII using `std::shared_ptr`.
