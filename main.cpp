#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <bitset>
#include <fstream>
#include "HNSW.h"
#include "ImageProcessor.h"

// Generate a dummy image file of random color blocks
void createDummyImage(const std::string& filepath, int width, int height) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return;
    
    // Write simple PPM header
    out << "P6\n" << width << " " << height << "\n255\n";
    
    // Very simple pseudo-random generation to make slightly different images
    std::mt19937 rng(std::hash<std::string>{}(filepath)); // deterministic random based on name
    std::uniform_int_distribution<int> colorDist(0, 255);
    
    for (int i = 0; i < width * height; ++i) {
        // Red, Green, Blue
        out << static_cast<unsigned char>(colorDist(rng))
            << static_cast<unsigned char>(colorDist(rng))
            << static_cast<unsigned char>(colorDist(rng));
    }
}

int main() {
    std::cout << "--- HNSW Reverse Image Search Test ---\n";

    // 1. Create a couple of dummy images
    std::cout << "[1] Generating dummy images...\n";
    createDummyImage("dummy1.ppm", 64, 64);
    createDummyImage("dummy2.ppm", 64, 64);

    // 2. Generate Hashes
    std::cout << "[2] Computing pHashes...\n";
    uint64_t hash1 = 0;
    uint64_t hash2 = 0;
    try {
        hash1 = ImageProcessor::generatePHash("dummy1.ppm");
        hash2 = ImageProcessor::generatePHash("dummy2.ppm");
        std::cout << "Hash 1: " << std::bitset<64>(hash1) << " (0x" << std::hex << hash1 << std::dec << ")\n";
        std::cout << "Hash 2: " << std::bitset<64>(hash2) << " (0x" << std::hex << hash2 << std::dec << ")\n";
        std::cout << "Hamming Distance between them: " << computeHammingDistance(hash1, hash2) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error generating hash: " << e.what() << "\n";
        return 1;
    }

    // 3. Populate HNSW Graph
    std::cout << "\n[3] Building HNSW Grapt with synthetic data...\n";
    HNSWIndex index(16, 32, 100);
    
    std::mt19937_64 rng64(42);
    int numNodes = 10000;
    std::vector<uint64_t> ground_truth_hashes;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numNodes; ++i) {
        uint64_t rndHash = rng64();
        index.insert(i, rndHash);
        ground_truth_hashes.push_back(rndHash);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Inserted " << index.size() << " nodes in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms.\n";

    // 4. Test Search K-NN
    std::cout << "\n[4] Querying HNSW Graph...\n";
    uint64_t queryHash = rng64();
    
    // True K-NN (Linear O(N) Search)
    t1 = std::chrono::high_resolution_clock::now();
    std::priority_queue<std::pair<uint32_t, uint64_t>> linearTopK;
    for (int i = 0; i < numNodes; ++i) {
        uint32_t dist = computeHammingDistance(queryHash, ground_truth_hashes[i]);
        if (linearTopK.size() < 5 || dist < linearTopK.top().first) {
            linearTopK.push({dist, i});
            if (linearTopK.size() > 5) linearTopK.pop();
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Linear search took " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us.\n";
    
    // HNSW Search
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> hnswResults = index.search(queryHash, 5, 50);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "HNSW search took " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us.\n\n";

    // Compare
    std::cout << "Linear Results (Dist -> ID):\n";
    while(!linearTopK.empty()){
        std::cout << "  " << linearTopK.top().first << " -> ID " << linearTopK.top().second << "\n";
        linearTopK.pop();
    }
    
    std::cout << "\nHNSW Results (Dist -> ID):\n";
    for(uint64_t resId : hnswResults) {
        std::cout << "  " << computeHammingDistance(queryHash, ground_truth_hashes[resId]) << " -> ID " << resId << "\n";
    }

    std::cout << "\nTest Complete.\n";
    return 0;
}
