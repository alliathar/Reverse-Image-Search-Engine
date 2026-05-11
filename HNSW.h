#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <queue>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <string>

using Embedding = std::vector<float>;

// Cosine distance over L2-normalized vectors: 1 - dot(a, b). Range [0, 2].
inline float computeDistance(const Embedding& a, const Embedding& b) {
    float dot = 0.0f;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) dot += a[i] * b[i];
    return 1.0f - dot;
}

struct HNSWNode {
    uint64_t id;
    Embedding hash;
    int maxLayer;
    std::vector<std::vector<uint64_t>> neighbors;

    HNSWNode(uint64_t _id, Embedding _hash, int _maxLayer)
        : id(_id), hash(std::move(_hash)), maxLayer(_maxLayer) {
        neighbors.resize(maxLayer + 1);
    }
};

class HNSWIndex {
public:
    HNSWIndex(int M = 16, int M_max0 = 32, int efConstruction = 100);

    void insert(uint64_t id, Embedding hash);

    std::vector<uint64_t> search(const Embedding& queryHash, int k, int efSearch = 50);

    std::string exportGraphJSON(const std::unordered_map<uint64_t, std::string>& idToPath) const;

    size_t size() const { return nodes_.size(); }

    const Embedding& getEmbedding(uint64_t id) const { return nodes_.at(id)->hash; }

private:
    int M_;
    int M_max0_;
    int efConstruction_;
    double levelMult_;

    uint64_t entryPointId_;
    int maxCurrentLayer_;
    bool hasEntryPoint_;

    std::unordered_map<uint64_t, std::shared_ptr<HNSWNode>> nodes_;
    std::mt19937 rng_;

    int generateRandomLayer();

    std::priority_queue<std::pair<float, uint64_t>> searchLayer(
        const Embedding& queryHash,
        uint64_t entryPoint,
        int ef,
        int layer
    );

    void selectNeighbors(std::vector<uint64_t>& neighbors,
                         std::priority_queue<std::pair<float, uint64_t>>& candidates,
                         int M);
};
