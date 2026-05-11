#include "HNSW.h"
#include <algorithm>
#include <limits>
#include <set>
#include <sstream>

template<typename T, typename DistanceFn>
HNSWIndex<T, DistanceFn>::HNSWIndex(int M, int M_max0, int efConstruction, DistanceFn dist)
    : M_(M), M_max0_(M_max0), efConstruction_(efConstruction),
      dist_(std::move(dist)),
      entryPointId_(0), maxCurrentLayer_(-1), hasEntryPoint_(false) {
    levelMult_ = 1.0 / std::log(static_cast<double>(M_));
    std::random_device rd;
    rng_.seed(rd());
}

template<typename T, typename DistanceFn>
int HNSWIndex<T, DistanceFn>::generateRandomLayer() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(rng_)) * levelMult_;
    return static_cast<int>(r);
}

template<typename T, typename DistanceFn>
void HNSWIndex<T, DistanceFn>::selectNeighbors(std::vector<uint64_t>& neighbors,
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

template<typename T, typename DistanceFn>
std::priority_queue<std::pair<float, uint64_t>>
HNSWIndex<T, DistanceFn>::searchLayer(const T& queryHash,
                                      uint64_t entryPoint,
                                      int ef,
                                      int layer) {
    std::priority_queue<std::pair<float, uint64_t>> topResults; // max-heap by distance
    std::priority_queue<std::pair<float, uint64_t>,
                        std::vector<std::pair<float, uint64_t>>,
                        std::greater<>> candidates; // min-heap
    std::set<uint64_t> visited;

    auto epNode = nodes_[entryPoint];
    float dist = dist_(queryHash, epNode->hash);
    bool  epDel = deleted_.count(entryPoint) > 0;

    // Always add the entry point to candidates — we need to traverse through
    // it regardless of its deletion status to reach live nodes nearby.
    candidates.emplace(dist, entryPoint);
    visited.insert(entryPoint);

    // Only surface the entry point as a result if it is not deleted.
    if (!epDel) {
        topResults.emplace(dist, entryPoint);
    }

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        // If the nearest unvisited candidate is already further than the
        // furthest element in our result set, we cannot improve — stop.
        // Guard against an empty topResults (possible when every node seen so
        // far is deleted): treat lowerBound as +inf in that case so we keep
        // searching.
        float lowerBound = topResults.empty()
                         ? std::numeric_limits<float>::max()
                         : topResults.top().first;

        if (current.first > lowerBound) {
            break;
        }

        auto currentNode = nodes_[current.second];
        if (layer > currentNode->maxLayer) continue;

        for (uint64_t neighborId : currentNode->neighbors[layer]) {
            if (visited.find(neighborId) == visited.end()) {
                visited.insert(neighborId);
                auto neighborNode  = nodes_[neighborId];
                float neighborDist = dist_(queryHash, neighborNode->hash);
                bool  neighborDel  = deleted_.count(neighborId) > 0;

                float worstResult = topResults.empty()
                                  ? std::numeric_limits<float>::max()
                                  : topResults.top().first;

                if (static_cast<int>(topResults.size()) < ef
                        || neighborDist < worstResult) {

                    // Traverse through this node regardless of deletion status
                    // so we can reach live nodes on the other side of it.
                    candidates.emplace(neighborDist, neighborId);

                    // Only add to results if the node is alive.
                    if (!neighborDel) {
                        topResults.emplace(neighborDist, neighborId);
                        if (static_cast<int>(topResults.size()) > ef) {
                            topResults.pop();
                        }
                    }
                }
            }
        }
    }

    return topResults;
}

template<typename T, typename DistanceFn>
void HNSWIndex<T, DistanceFn>::insert(uint64_t id, T hash) {
    if (nodes_.find(id) != nodes_.end()) {
        return;
    }

    int layer = generateRandomLayer();
    auto newNode = std::make_shared<HNSWNode<T>>(id, std::move(hash), layer);
    nodes_[id] = newNode;
    const T& nodeHash = newNode->hash;

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
            float minDist = dist_(nodeHash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                float d = dist_(nodeHash, nodes_[neighborId]->hash);
                if (d < minDist) {
                    minDist = d;
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
                    float d = dist_(neighborNode->hash, nodes_[nId]->hash);
                    nCandidates.emplace(d, nId);
                }
                selectNeighbors(neighborNode->neighbors[lc], nCandidates, M_max);
            }
        }

        if (lc > 0) {
            float minDist = std::numeric_limits<float>::max();
            for (uint64_t nId : newNode->neighbors[lc]) {
                float d = dist_(nodeHash, nodes_[nId]->hash);
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

// ---------------------------------------------------------------------------
// softDelete
//
// Adds the id to the deleted_ tombstone set.  The node stays in nodes_ so
// the graph topology is preserved — removing it would leave dangling neighbor
// references and break traversal.
//
// Entry-point edge case: if we delete the current entry point we walk its
// neighbor lists (from the highest layer downward) to find the first live
// replacement.  If no live node exists at all (every node was deleted), we
// set hasEntryPoint_ = false so subsequent searches return empty immediately.
// ---------------------------------------------------------------------------
template<typename T, typename DistanceFn>
void HNSWIndex<T, DistanceFn>::softDelete(uint64_t id) {
    // Unknown id — nothing to do.
    if (nodes_.find(id) == nodes_.end()) return;

    // Already deleted — idempotent, don't double-count.
    if (deleted_.count(id)) return;

    deleted_.insert(id);

    // If we just tombstoned the entry point, find a live replacement so
    // future searches have a valid starting node.
    if (id == entryPointId_) {
        bool replaced = false;

        // Walk from the highest layer downward for the best chance of finding
        // a well-connected live node quickly.
        for (int lc = maxCurrentLayer_; lc >= 0 && !replaced; --lc) {
            for (uint64_t nbr : nodes_[id]->neighbors[lc]) {
                if (!deleted_.count(nbr)) {
                    entryPointId_ = nbr;
                    replaced      = true;
                    break;
                }
            }
        }

        if (!replaced) {
            // Every node reachable from the old entry point is also deleted.
            // The index is effectively empty for search purposes.
            hasEntryPoint_ = false;
        }
    }
}

template<typename T, typename DistanceFn>
void HNSWIndex<T, DistanceFn>::updatePoint(uint64_t id, T newHash) {
    if (nodes_.find(id) == nodes_.end()) return;

    // Scrub stale back-references from neighbors before removing the node
    auto oldNode = nodes_[id];
    for (int lc = 0; lc <= oldNode->maxLayer; ++lc) {
        for (uint64_t nbr : oldNode->neighbors[lc]) {
            if (nodes_.find(nbr) == nodes_.end()) continue;
            auto& nbrList = nodes_[nbr]->neighbors[lc];
            nbrList.erase(
                std::remove(nbrList.begin(), nbrList.end(), id),
                nbrList.end()
            );
        }
    }

    // Tombstone first so softDelete can fix the entry point if needed
    softDelete(id);

    // Now safely erase — softDelete has already updated entryPointId_ if needed
    nodes_.erase(id);
    deleted_.erase(id);

    // Re-insert with new hash as a completely fresh node
    insert(id, std::move(newHash));
}

template<typename T, typename DistanceFn>
std::vector<uint64_t> HNSWIndex<T, DistanceFn>::search(const T& queryHash, int k, int efSearch) {
    std::vector<uint64_t> result;
    if (!hasEntryPoint_) return result;

    efSearch = std::max(efSearch, k);
    uint64_t currentEp = entryPointId_;

    // Greedy descent: still traverse deleted nodes at upper layers because
    // they may be the only bridge to the correct neighbourhood.
    for (int lc = maxCurrentLayer_; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            float minDist = dist_(queryHash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                float d = dist_(queryHash, nodes_[neighborId]->hash);
                if (d < minDist) {
                    minDist = d;
                    bestEp  = neighborId;
                    changed = true;
                }
            }
            currentEp = bestEp;
        }
    }

    auto topResults = searchLayer(queryHash, currentEp, efSearch, 0);

    // Trim to k.
    while (static_cast<int>(topResults.size()) > k) {
        topResults.pop();
    }

    result.reserve(topResults.size());
    while (!topResults.empty()) {
        result.push_back(topResults.top().second);
        topResults.pop();
    }

    // searchLayer already filters deleted nodes from topResults, but the
    // greedy entry-point descent at upper layers may have landed on a deleted
    // node that then became currentEp for layer-0 search.  That is safe
    // because searchLayer handles a deleted entry point correctly.  This
    // final pass is a belt-and-suspenders guard — it costs O(k) and
    // catches any future code paths that might bypass searchLayer's filter.
    result.erase(
        std::remove_if(result.begin(), result.end(),
                       [this](uint64_t id){ return deleted_.count(id) > 0; }),
        result.end()
    );

    std::reverse(result.begin(), result.end()); // closest first
    return result;
}

template<typename T, typename DistanceFn>
std::priority_queue<std::pair<float, uint64_t>,
                    std::vector<std::pair<float, uint64_t>>,
                    std::greater<>>
HNSWIndex<T, DistanceFn>::searchLayerFurthest(const T& queryHash,
                                              uint64_t entryPoint,
                                              int ef,
                                              int layer) {
    // topResults: min-heap by distance — the "worst" kept is the closest one.
    std::priority_queue<std::pair<float, uint64_t>,
                        std::vector<std::pair<float, uint64_t>>,
                        std::greater<>> topResults;
    // candidates: max-heap — explore the furthest unexplored next.
    std::priority_queue<std::pair<float, uint64_t>> candidates;
    std::set<uint64_t> visited;

    auto epNode = nodes_[entryPoint];
    float dist = dist_(queryHash, epNode->hash);

    topResults.emplace(dist, entryPoint);
    candidates.emplace(dist, entryPoint);
    visited.insert(entryPoint);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        float lowerBound = topResults.top().first;  // closest of our best
        if (current.first < lowerBound) {
            // The furthest unexplored is already closer than our worst kept;
            // exploring it can't push us further away from the query.
            break;
        }

        auto currentNode = nodes_[current.second];
        if (layer > currentNode->maxLayer) continue;

        for (uint64_t neighborId : currentNode->neighbors[layer]) {
            if (visited.find(neighborId) == visited.end()) {
                visited.insert(neighborId);
                auto neighborNode = nodes_[neighborId];
                float neighborDist = dist_(queryHash, neighborNode->hash);

                if (static_cast<int>(topResults.size()) < ef || neighborDist > topResults.top().first) {
                    candidates.emplace(neighborDist, neighborId);
                    topResults.emplace(neighborDist, neighborId);
                    if (static_cast<int>(topResults.size()) > ef) {
                        topResults.pop();  // drop the closest (worst in furthest-mode)
                    }
                }
            }
        }
    }

    return topResults;
}

template<typename T, typename DistanceFn>
std::vector<uint64_t> HNSWIndex<T, DistanceFn>::searchFurthest(const T& queryHash, int k, int efSearch) {
    std::vector<uint64_t> result;
    if (!hasEntryPoint_) return result;

    efSearch = std::max(efSearch, k);
    uint64_t currentEp = entryPointId_;

    // Phase 1: greedy ascent through layers — move to the FURTHEST neighbor at each step.
    for (int lc = maxCurrentLayer_; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            float maxDist = dist_(queryHash, nodes_[currentEp]->hash);
            uint64_t bestEp = currentEp;

            for (uint64_t neighborId : nodes_[currentEp]->neighbors[lc]) {
                float d = dist_(queryHash, nodes_[neighborId]->hash);
                if (d > maxDist) {
                    maxDist = d;
                    bestEp = neighborId;
                    changed = true;
                }
            }
            currentEp = bestEp;
        }
    }

    auto topResults = searchLayerFurthest(queryHash, currentEp, efSearch, 0);

    while (static_cast<int>(topResults.size()) > k) {
        topResults.pop();  // drop closest until k remain
    }

    // Pop from min-heap gives closest-first; reverse to put furthest first.
    result.reserve(topResults.size());
    while (!topResults.empty()) {
        result.push_back(topResults.top().second);
        topResults.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::string escapeJSONString(const std::string& input) {
    std::string output;
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
// ---------------------------------------------------------------------------
// rebuildFrom — free function, not a member, because it needs to construct a
// fresh index.  It simply re-inserts every live node from `old` into a new
// index with the same hyperparameters.
//
// Insertion order is unspecified (unordered_map iteration order), which means
// the rebuilt graph's layer assignments will differ from the original.  That
// is intentional and expected — ANN quality typically improves because the
// graph is no longer biased by edges that used to connect to deleted nodes.
// ---------------------------------------------------------------------------
template<typename T, typename DistanceFn>
HNSWIndex<T, DistanceFn> rebuildFrom(const HNSWIndex<T, DistanceFn>& old) {
    HNSWIndex<T, DistanceFn> fresh(
        old.getM(),
        old.getM_max0(),
        old.getEfConstruction(),
        DistanceFn{}
    );

    for (uint64_t id : old.getAllIds()) {
        if (!old.isDeleted(id)) {
            fresh.insert(id, old.getEmbedding(id));
        }
    }
    return fresh;
}


template<typename T, typename DistanceFn>
std::string HNSWIndex<T, DistanceFn>::exportGraphJSON(
        const std::unordered_map<uint64_t, std::string>& idToPath) const {
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

// Explicit template instantiations — exactly these two flavors are usable.
template class HNSWIndex<uint64_t, HammingDistance>;
template class HNSWIndex<std::vector<float>, CosineDistance>;


// Explicit instantiations for the free rebuild function.
template HNSWIndex<uint64_t, HammingDistance>
    rebuildFrom(const HNSWIndex<uint64_t, HammingDistance>&);
template HNSWIndex<std::vector<float>, CosineDistance>
    rebuildFrom(const HNSWIndex<std::vector<float>, CosineDistance>&);
