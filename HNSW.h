#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <queue>
#include <unordered_map>
#include <memory>
#include <cmath>

// O(1) Hamming distance for 64-bit integers
#ifdef _MSC_VER
#include <intrin.h>
#endif

inline uint32_t computeHammingDistance(uint64_t a, uint64_t b) {
#ifdef _MSC_VER
    return static_cast<uint32_t>(__popcnt64(a ^ b));
#else
    return __builtin_popcountll(a ^ b);
#endif
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

    // Get number of nodes
    size_t size() const { return nodes_.size(); }

private:
    int M_;
    int M_max0_;
    int efConstruction_;
    double levelMult_;

    uint64_t entryPointId_;
    int maxCurrentLayer_;
    bool hasEntryPoint_;
    
    // Maps unique image IDs to graph nodes (RAII managed)
    std::unordered_map<uint64_t, std::shared_ptr<HNSWNode>> nodes_;
    std::mt19937 rng_;

    int generateRandomLayer();
    
    // Greedy search for HNSW within a single layer returning closest neighbors
    std::priority_queue<std::pair<uint32_t, uint64_t>> searchLayer(
        uint64_t queryHash,
        uint64_t entryPoint,
        int ef,
        int layer
    );

    // Select neighbors heuristically (simple version: just takes closest)
    void selectNeighbors(std::vector<uint64_t>& neighbors, std::priority_queue<std::pair<uint32_t, uint64_t>>& candidates, int M);
};
