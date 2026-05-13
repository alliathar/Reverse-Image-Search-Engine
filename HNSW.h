#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cmath>
#include <string>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// O(1) Hamming distance over 64-bit pHash values, returned as float for uniformity.
struct HammingDistance {
    float operator()(uint64_t a, uint64_t b) const {
#ifdef _MSC_VER
        return static_cast<float>(__popcnt64(a ^ b));
#else
        return static_cast<float>(__builtin_popcountll(a ^ b));
#endif
    }
};

// Cosine distance for L2-normalized embeddings: 1 - dot(a, b). Range [0, 2].
struct CosineDistance {
    float operator()(const std::vector<float>& a, const std::vector<float>& b) const {
        float dot = 0.0f;
        size_t n = a.size();
        for (size_t i = 0; i < n; ++i) dot += a[i] * b[i];
        return 1.0f - dot;
    }
};

template<typename T>
struct HNSWNode {
    uint64_t id;
    T hash;
    int maxLayer;
    std::vector<std::vector<uint64_t>> neighbors;

    HNSWNode(uint64_t _id, T _hash, int _maxLayer)
        : id(_id), hash(std::move(_hash)), maxLayer(_maxLayer) {
        neighbors.resize(maxLayer + 1);
    }
};

template<typename T, typename DistanceFn>
class HNSWIndex {
public:
    HNSWIndex(int M = 16, int M_max0 = 32, int efConstruction = 100, DistanceFn dist = DistanceFn{});

    void insert(uint64_t id, T hash);
    void   softDelete(uint64_t id);
    void updatePoint(uint64_t id, T newHash);
    bool   isDeleted(uint64_t id)  const { return deleted_.count(id) > 0; }
    size_t deletedCount()          const { return deleted_.size(); }

    std::vector<uint64_t> search(const T& queryHash, int k, int efSearch = 50);
    // Reverse-greedy traversal: walks to the FURTHEST neighbor at each step.
    // Returns a local maximum of distance — not guaranteed to be the global
    // furthest point in the dataset, because the graph is nearest-neighbor.
    std::vector<uint64_t> searchFurthest(const T& queryHash, int k, int efSearch = 50);
    std::string exportGraphJSON(const std::unordered_map<uint64_t, std::string>& idToPath) const;
    size_t size() const { return nodes_.size(); }
    int getM()              const { return M_; }
    int getM_max0()         const { return M_max0_; }
    int getEfConstruction() const { return efConstruction_; }
    const T& getEmbedding(uint64_t id) const { return nodes_.at(id)->hash; }
    std::vector<uint64_t> getAllIds() const {
        std::vector<uint64_t> ids;
        ids.reserve(nodes_.size());
        for (const auto& p : nodes_) ids.push_back(p.first);
        return ids;
    }

private:
    int M_;
    int M_max0_;
    int efConstruction_;
    double levelMult_;
    DistanceFn dist_;

    uint64_t entryPointId_;
    int maxCurrentLayer_;
    bool hasEntryPoint_;

    std::unordered_map<uint64_t, std::shared_ptr<HNSWNode<T>>> nodes_;
    std::unordered_set<uint64_t> deleted_;
    std::mt19937 rng_;

    int generateRandomLayer();

    std::priority_queue<std::pair<float, uint64_t>> searchLayer(
        const T& queryHash,
        uint64_t entryPoint,
        int ef,
        int layer
    );

    // Furthest-mode variant of searchLayer. Returns a min-heap (closest at top)
    // so the caller can pop the closest excess down to k after the search.
    std::priority_queue<std::pair<float, uint64_t>,
                        std::vector<std::pair<float, uint64_t>>,
                        std::greater<>>
    searchLayerFurthest(
        const T& queryHash,
        uint64_t entryPoint,
        int ef,
        int layer
    );
    
    void selectNeighbors(std::vector<uint64_t>& neighbors,
                         std::priority_queue<std::pair<float, uint64_t>>& candidates,
                         int M);
};

using PHashHNSW     = HNSWIndex<uint64_t, HammingDistance>;
using EmbeddingHNSW = HNSWIndex<std::vector<float>, CosineDistance>;

// JSON escaping helper, defined in HNSW.cpp.
std::string escapeJSONString(const std::string& input);
