#include "HNSW.h"
#include <algorithm>
#include <set>

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

void HNSWIndex::selectNeighbors(std::vector<uint64_t>& neighbors, std::priority_queue<std::pair<uint32_t, uint64_t>>& candidates, int M) {
    // A simple heuristic: take the closest M candidates
    while (candidates.size() > M) {
        candidates.pop(); // priority queue top is the *furthest* element (max heap by distance)
    }
    neighbors.clear();
    while (!candidates.empty()) {
        neighbors.push_back(candidates.top().second);
        candidates.pop();
    }
}

std::priority_queue<std::pair<uint32_t, uint64_t>> HNSWIndex::searchLayer(
    uint64_t queryHash,
    uint64_t entryPoint,
    int ef,
    int layer) 
{
    // max heap, top is the furthest found candidate among the nearest
    std::priority_queue<std::pair<uint32_t, uint64_t>> topResults;
    // min heap, top is the closest candidate to explore
    std::priority_queue<std::pair<uint32_t, uint64_t>, std::vector<std::pair<uint32_t, uint64_t>>, std::greater<>> candidates;
    
    std::set<uint64_t> visited;

    auto epNode = nodes_[entryPoint];
    uint32_t dist = computeHammingDistance(queryHash, epNode->hash);

    topResults.emplace(dist, entryPoint);
    candidates.emplace(dist, entryPoint);
    visited.insert(entryPoint);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        uint32_t lowerBound = topResults.top().first;
        if (current.first > lowerBound) {
            break; // furthest candidate is closer than closest unexplored
        }

        auto currentNode = nodes_[current.second];
        if (layer > currentNode->maxLayer) continue;

        for (uint64_t neighborId : currentNode->neighbors[layer]) {
            if (visited.find(neighborId) == visited.end()) {
                visited.insert(neighborId);
                auto neighborNode = nodes_[neighborId];
                uint32_t neighborDist = computeHammingDistance(queryHash, neighborNode->hash);

                if (topResults.size() < ef || neighborDist < topResults.top().first) {
                    candidates.emplace(neighborDist, neighborId);
                    topResults.emplace(neighborDist, neighborId);

                    if (topResults.size() > ef) {
                        topResults.pop();
                    }
                }
            }
        }
    }

    return topResults;
}

void HNSWIndex::insert(uint64_t id, uint64_t hash) {
    if (nodes_.find(id) != nodes_.end()) {
        return; // id already exists
    }

    int layer = generateRandomLayer();
    auto newNode = std::make_shared<HNSWNode>(id, hash, layer);
    nodes_[id] = newNode;

    if (!hasEntryPoint_) {
        entryPointId_ = id;
        maxCurrentLayer_ = layer;
        hasEntryPoint_ = true;
        return;
    }

    uint64_t currentEp = entryPointId_;
    uint64_t currentEpHash = nodes_[currentEp]->hash;
    int currentMaxLayer = maxCurrentLayer_;

    // Phase 1: search for best entry point from top layer down to max(layer + 1, 0)
    for (int lc = currentMaxLayer; lc > layer; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            uint32_t minDist = computeHammingDistance(hash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                uint32_t dist = computeHammingDistance(hash, nodes_[neighborId]->hash);
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

    // candidates holds elements to connect to
    for (int lc = minLayer; lc >= 0; --lc) {
        auto topResults = searchLayer(hash, currentEp, efConstruction_, lc);
        
        // Select neighbors for the new node
        int M_max = (lc == 0) ? M_max0_ : M_;
        selectNeighbors(newNode->neighbors[lc], topResults, M_);

        // Add mutual connections
        for (uint64_t neighborId : newNode->neighbors[lc]) {
            auto neighborNode = nodes_[neighborId];
            neighborNode->neighbors[lc].push_back(id);
            
            // Shrink if capacity exceeded
            if (neighborNode->neighbors[lc].size() > M_max) {
                // Heuristic: just re-evaluate neighbors with priority queue
                std::priority_queue<std::pair<uint32_t, uint64_t>> nCandidates;
                for (uint64_t nId : neighborNode->neighbors[lc]) {
                    uint32_t d = computeHammingDistance(neighborNode->hash, nodes_[nId]->hash);
                    nCandidates.emplace(d, nId);
                }
                selectNeighbors(neighborNode->neighbors[lc], nCandidates, M_max);
            }
        }
        
        if (lc > 0) { // Search for next layer entry point
            uint32_t minDist = static_cast<uint32_t>(-1);
            for (uint64_t nId : newNode->neighbors[lc]) {
                uint32_t d = computeHammingDistance(hash, nodes_[nId]->hash);
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

std::vector<uint64_t> HNSWIndex::search(uint64_t queryHash, int k, int efSearch) {
    std::vector<uint64_t> result;
    if (!hasEntryPoint_) return result;

    efSearch = std::max(efSearch, k);
    uint64_t currentEp = entryPointId_;

    // Phase 1: fast search to layer 1
    for (int lc = maxCurrentLayer_; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            uint32_t minDist = computeHammingDistance(queryHash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                uint32_t dist = computeHammingDistance(queryHash, nodes_[neighborId]->hash);
                if (dist < minDist) {
                    minDist = dist;
                    bestEp = neighborId;
                    changed = true;
                }
            }
            currentEp = bestEp;
        }
    }

    // Phase 2: detailed search at level 0
    auto topResults = searchLayer(queryHash, currentEp, efSearch, 0);

    while (topResults.size() > k) {
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

#include <sstream>

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
    ss << "{\n  \"nodes\": [\n";
    bool firstNode = true;
    for (const auto& pair : nodes_) {
        if (!firstNode) ss << ",\n";
        firstNode = false;
        
        uint64_t id = pair.first;
        auto node = pair.second;
        
        std::string path = idToPath.count(id) ? idToPath.at(id) : "unknown";
        ss << "    {\"id\": " << id << ", \"path\": \"" << escapeJSONString(path) << "\", \"layer\": " << node->maxLayer << "}";
    }
    ss << "\n  ],\n  \"edges\": [\n";
    
    bool firstEdge = true;
    for (const auto& pair : nodes_) {
        uint64_t sourceId = pair.first;
        auto node = pair.second;
        
        for (int layer = 0; layer <= node->maxLayer; ++layer) {
            for (uint64_t targetId : node->neighbors[layer]) {
                if (!firstEdge) ss << ",\n";
                firstEdge = false;
                ss << "    {\"source\": " << sourceId << ", \"target\": " << targetId << ", \"layer\": " << layer << "}";
            }
        }
    }
    ss << "\n  ]\n}";
    return ss.str();
}