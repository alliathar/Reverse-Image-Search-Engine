#include "HNSW.h"
#include <algorithm>
#include <limits>
#include <set>
#include <sstream>

HNSWIndex::HNSWIndex(int M, int M_max0, int efConstruction)
    : M_(M), M_max0_(M_max0), efConstruction_(efConstruction),
      maxCurrentLayer_(-1), hasEntryPoint_(false), entryPointId_(0) {
    levelMult_ = 1.0 / std::log(static_cast<double>(M_));
    std::random_device rd;
    rng_.seed(rd());
}

int HNSWIndex::generateRandomLayer() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(rng_)) * levelMult_;
    return static_cast<int>(r);
}

void HNSWIndex::selectNeighbors(std::vector<uint64_t>& neighbors,
                                std::priority_queue<std::pair<float, uint64_t>>& candidates,
                                int M) {
    while (static_cast<int>(candidates.size()) > M) {
        candidates.pop(); // max-heap: top is the furthest element
    }
    neighbors.clear();
    while (!candidates.empty()) {
        neighbors.push_back(candidates.top().second);
        candidates.pop();
    }
}

std::priority_queue<std::pair<float, uint64_t>> HNSWIndex::searchLayer(
    const Embedding& queryHash,
    uint64_t entryPoint,
    int ef,
    int layer)
{
    std::priority_queue<std::pair<float, uint64_t>> topResults; // max-heap by distance
    std::priority_queue<std::pair<float, uint64_t>,
                        std::vector<std::pair<float, uint64_t>>,
                        std::greater<>> candidates; // min-heap
    std::set<uint64_t> visited;

    auto epNode = nodes_[entryPoint];
    float dist = computeDistance(queryHash, epNode->hash);

    topResults.emplace(dist, entryPoint);
    candidates.emplace(dist, entryPoint);
    visited.insert(entryPoint);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        float lowerBound = topResults.top().first;
        if (current.first > lowerBound) {
            break;
        }

        auto currentNode = nodes_[current.second];
        if (layer > currentNode->maxLayer) continue;

        for (uint64_t neighborId : currentNode->neighbors[layer]) {
            if (visited.find(neighborId) == visited.end()) {
                visited.insert(neighborId);
                auto neighborNode = nodes_[neighborId];
                float neighborDist = computeDistance(queryHash, neighborNode->hash);

                if (static_cast<int>(topResults.size()) < ef || neighborDist < topResults.top().first) {
                    candidates.emplace(neighborDist, neighborId);
                    topResults.emplace(neighborDist, neighborId);
                    if (static_cast<int>(topResults.size()) > ef) {
                        topResults.pop();
                    }
                }
            }
        }
    }

    return topResults;
}

void HNSWIndex::insert(uint64_t id, Embedding hash) {
    if (nodes_.find(id) != nodes_.end()) {
        return;
    }

    int layer = generateRandomLayer();
    auto newNode = std::make_shared<HNSWNode>(id, std::move(hash), layer);
    nodes_[id] = newNode;
    const Embedding& nodeHash = newNode->hash;

    if (!hasEntryPoint_) {
        entryPointId_ = id;
        maxCurrentLayer_ = layer;
        hasEntryPoint_ = true;
        return;
    }

    uint64_t currentEp = entryPointId_;
    int currentMaxLayer = maxCurrentLayer_;

    // Phase 1: greedy search from top layer down to layer + 1
    for (int lc = currentMaxLayer; lc > layer; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            float minDist = computeDistance(nodeHash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                float dist = computeDistance(nodeHash, nodes_[neighborId]->hash);
                if (dist < minDist) {
                    minDist = dist;
                    bestEp = neighborId;
                    changed = true;
                }
            }
            currentEp = bestEp;
        }
    }

    // Phase 2: insert into all layers <= min(maxCurrentLayer, layer)
    int minLayer = std::min(layer, currentMaxLayer);

    for (int lc = minLayer; lc >= 0; --lc) {
        auto topResults = searchLayer(nodeHash, currentEp, efConstruction_, lc);

        int M_max = (lc == 0) ? M_max0_ : M_;
        selectNeighbors(newNode->neighbors[lc], topResults, M_);

        for (uint64_t neighborId : newNode->neighbors[lc]) {
            auto neighborNode = nodes_[neighborId];
            neighborNode->neighbors[lc].push_back(id);

            if (static_cast<int>(neighborNode->neighbors[lc].size()) > M_max) {
                std::priority_queue<std::pair<float, uint64_t>> nCandidates;
                for (uint64_t nId : neighborNode->neighbors[lc]) {
                    float d = computeDistance(neighborNode->hash, nodes_[nId]->hash);
                    nCandidates.emplace(d, nId);
                }
                selectNeighbors(neighborNode->neighbors[lc], nCandidates, M_max);
            }
        }

        if (lc > 0) {
            float minDist = std::numeric_limits<float>::max();
            for (uint64_t nId : newNode->neighbors[lc]) {
                float d = computeDistance(nodeHash, nodes_[nId]->hash);
                if (d < minDist) {
                    minDist = d;
                    currentEp = nId;
                }
            }
        }
    }

    if (layer > currentMaxLayer) {
        entryPointId_ = id;
        maxCurrentLayer_ = layer;
    }
}

std::vector<uint64_t> HNSWIndex::search(const Embedding& queryHash, int k, int efSearch) {
    std::vector<uint64_t> result;
    if (!hasEntryPoint_) return result;

    efSearch = std::max(efSearch, k);
    uint64_t currentEp = entryPointId_;

    for (int lc = maxCurrentLayer_; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            float minDist = computeDistance(queryHash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                float dist = computeDistance(queryHash, nodes_[neighborId]->hash);
                if (dist < minDist) {
                    minDist = dist;
                    bestEp = neighborId;
                    changed = true;
                }
            }
            currentEp = bestEp;
        }
    }

    auto topResults = searchLayer(queryHash, currentEp, efSearch, 0);

    while (static_cast<int>(topResults.size()) > k) {
        topResults.pop();
    }

    result.reserve(topResults.size());
    while (!topResults.empty()) {
        result.push_back(topResults.top().second);
        topResults.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

std::string escapeJSONString(const std::string& input) {
    std::string output = "";
    for (char c : input) {
        if (c == '"') output += "\\\"";
        else if (c == '\\') output += "\\\\";
        else if (c == '\b') output += "\\b";
        else if (c == '\f') output += "\\f";
        else if (c == '\n') output += "\\n";
        else if (c == '\r') output += "\\r";
        else if (c == '\t') output += "\\t";
        else output += c;
    }
    return output;
}

std::string HNSWIndex::exportGraphJSON(const std::unordered_map<uint64_t, std::string>& idToPath) const {
    std::stringstream ss;
    ss << "{\"nodes\":[";
    bool firstNode = true;
    for (const auto& pair : nodes_) {
        if (!firstNode) ss << ",";
        firstNode = false;

        uint64_t id = pair.first;
        auto node = pair.second;

        std::string path = idToPath.count(id) ? idToPath.at(id) : "unknown";
        ss << "{\"id\":" << id << ",\"path\":\"" << escapeJSONString(path) << "\",\"layer\":" << node->maxLayer << "}";
    }
    ss << "],\"edges\":[";

    bool firstEdge = true;
    for (const auto& pair : nodes_) {
        uint64_t sourceId = pair.first;
        auto node = pair.second;

        for (int layer = 0; layer <= node->maxLayer; ++layer) {
            for (uint64_t targetId : node->neighbors[layer]) {
                if (!firstEdge) ss << ",";
                firstEdge = false;
                ss << "{\"source\":" << sourceId << ",\"target\":" << targetId << ",\"layer\":" << layer << "}";
            }
        }
    }
    ss << "]}";
    return ss.str();
}
