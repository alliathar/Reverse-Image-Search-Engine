#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <sstream>
#include <cmath>
#include "HNSW.h"
#include "ImageProcessor.h"

namespace fs = std::filesystem;

// ----- Backend selection -----
// Define USE_PHASH at compile time (via CMake -DUSE_PHASH=ON) to use the legacy
// 64-bit DCT pHash with Hamming distance. Default is the CNN embedding backend.
#ifdef USE_PHASH
    using HashT = uint64_t;
    using IndexT = PHashHNSW;
    using DistanceFn = HammingDistance;
    static HashT extractHash(const std::string& path) {
        return ImageProcessor::generatePHash(path);
    }
    // Bit-wise majority vote across N 64-bit pHashes.
    static HashT combineHashes(const std::vector<HashT>& hashes) {
        std::vector<int> bitCounts(64, 0);
        for (HashT h : hashes) {
            for (int i = 0; i < 64; ++i) {
                if (h & (1ULL << i)) bitCounts[i]++;
            }
        }
        HashT combined = 0;
        int threshold = static_cast<int>(hashes.size() + 1) / 2;
        for (int i = 0; i < 64; ++i) {
            if (bitCounts[i] >= threshold) combined |= (1ULL << i);
        }
        return combined;
    }
#else
    using HashT = std::vector<float>;
    using IndexT = EmbeddingHNSW;
    using DistanceFn = CosineDistance;
    static HashT extractHash(const std::string& path) {
        return ImageProcessor::generateEmbedding(path);
    }
    // Element-wise mean of L2-normalized embeddings, then L2-normalize again.
    static HashT combineHashes(const std::vector<HashT>& hashes) {
        if (hashes.empty()) return {};
        HashT combined(hashes[0].size(), 0.0f);
        for (const auto& h : hashes) {
            for (size_t i = 0; i < combined.size(); ++i) combined[i] += h[i];
        }
        double norm_sq = 0.0;
        for (float v : combined) norm_sq += v * v;
        float norm = static_cast<float>(std::sqrt(norm_sq));
        if (norm > 1e-12f) for (float& v : combined) v /= norm;
        return combined;
    }
#endif

int main(int /*argc*/, char* argv[]) {
#ifndef USE_PHASH
    fs::path exeDir = fs::weakly_canonical(fs::path(argv[0])).parent_path();
    fs::path modelPath = exeDir / ".." / "models" / "mobilenetv3_small.onnx";
    try {
        ImageProcessor::initialize(modelPath.string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX model at " << modelPath << ": " << e.what() << std::endl;
        return 1;
    }
#else
    (void)argv;
#endif

    std::unordered_map<uint64_t, std::string> idToPath;
    std::unordered_map<uint64_t, std::string> idToCategory;
    std::unordered_map<std::string, std::shared_ptr<IndexT>> categoryIndexes;
    uint64_t nextId = 0;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        std::string word;
        std::vector<std::string> words;
        while (ss >> word) words.push_back(word);
        if (words.empty()) continue;

        if (words[0] == "LOAD") {
            idToPath.clear();
            idToCategory.clear();
            categoryIndexes.clear();
            nextId = 0;

            std::string datasetPath;
            for (size_t i = 1; i < words.size(); i++) {
                if (i > 1) datasetPath += " ";
                datasetPath += words[i];
            }

            try {
                for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
                    if (!entry.is_regular_file()) continue;
                    std::string pathStr = entry.path().string();
                    std::string category = entry.path().parent_path().filename().string();

                    try {
                        HashT hash = extractHash(pathStr);
                        idToPath[nextId] = pathStr;
                        idToCategory[nextId] = category;

                        if (categoryIndexes.find(category) == categoryIndexes.end()) {
                            categoryIndexes[category] = std::make_shared<IndexT>(16, 32, 100);
                        }
                        categoryIndexes[category]->insert(nextId, std::move(hash));
                        nextId++;
                    } catch (...) {
                        // Skip non-images or unreadable files
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "{\"error\":\"" << escapeJSONString(e.what()) << "\"}\n" << std::flush;
                continue;
            }

            std::string graphJSON = "{}";
            if (!categoryIndexes.empty()) {
                graphJSON = categoryIndexes.begin()->second->exportGraphJSON(idToPath);
            }

            // Collect category names sorted for the frontend dropdown.
            std::vector<std::string> categoryNames;
            categoryNames.reserve(categoryIndexes.size());
            for (const auto& p : categoryIndexes) categoryNames.push_back(p.first);
            std::sort(categoryNames.begin(), categoryNames.end());

            std::ostringstream out;
            out << "{\"status\":\"ready\",\"count\":" << nextId << ",\"categories\":[";
            for (size_t i = 0; i < categoryNames.size(); ++i) {
                if (i > 0) out << ",";
                out << "\"" << escapeJSONString(categoryNames[i]) << "\"";
            }
            out << "],\"graph\":" << graphJSON << "}";
            std::cout << out.str() << "\n" << std::flush;

        } else if (words[0] == "SEARCH") {
            // Protocol: SEARCH <mode> <category> <path1> [<path2>...]
            //   mode:     "normal" | "negative" | "multi"
            //   category: "all" or specific category name
            if (words.size() < 4) {
                std::cout << "{\"error\":\"Usage: SEARCH <mode> <category> <path>...\"}\n" << std::flush;
                continue;
            }
            std::string mode = words[1];
            std::string targetCategory = words[2];
            std::vector<std::string> queryPaths(words.begin() + 3, words.end());

            // Compute / combine query hashes.
            HashT queryHash;
            try {
                if (mode == "multi" && queryPaths.size() > 1) {
                    std::vector<HashT> hashes;
                    hashes.reserve(queryPaths.size());
                    for (const auto& p : queryPaths) hashes.push_back(extractHash(p));
                    queryHash = combineHashes(hashes);
                } else {
                    queryHash = extractHash(queryPaths[0]);
                }
            } catch (...) {
                std::cout << "{\"error\":\"Could not read query image\"}\n" << std::flush;
                continue;
            }

            struct Match {
                uint64_t id;
                float distance;
            };
            std::vector<Match> allMatches;
            DistanceFn dfn;

            // Resolve which indexes to scan.
            std::vector<std::shared_ptr<IndexT>> targetIndexes;
            if (targetCategory == "all" || targetCategory == "") {
                for (auto& p : categoryIndexes) targetIndexes.push_back(p.second);
            } else {
                auto it = categoryIndexes.find(targetCategory);
                if (it == categoryIndexes.end()) {
                    std::cout << "{\"error\":\"Category not found in dataset\"}\n" << std::flush;
                    continue;
                }
                targetIndexes.push_back(it->second);
            }

            if (mode == "negative") {
                // Linear scan: return the FURTHEST images. HNSW is built for "find closest"
                // and can't traverse "furthest" cheaply, so we score everything.
                for (auto& idx : targetIndexes) {
                    for (uint64_t resId : idx->getAllIds()) {
                        allMatches.push_back({resId, dfn(queryHash, idx->getEmbedding(resId))});
                    }
                }
                // Sort by distance descending (furthest first).
                std::sort(allMatches.begin(), allMatches.end(),
                          [](const Match& a, const Match& b) { return a.distance > b.distance; });
            } else {
                // normal / multi (multi just combined the embeddings above) — HNSW nearest neighbor search.
                for (auto& idx : targetIndexes) {
                    auto results = idx->search(queryHash, 12, 50);
                    for (uint64_t resId : results) {
                        allMatches.push_back({resId, dfn(queryHash, idx->getEmbedding(resId))});
                    }
                }
                std::sort(allMatches.begin(), allMatches.end(),
                          [](const Match& a, const Match& b) { return a.distance < b.distance; });
            }

            if (allMatches.size() > 12) allMatches.resize(12);

            std::ostringstream out;
            out << "{\"results\":[";
            bool first = true;
            for (const auto& match : allMatches) {
                if (!first) out << ",";
                first = false;

#ifdef USE_PHASH
                // Hamming distance over 64 bits: 0 = identical, 64 = opposite.
                double accuracy = ((64.0 - static_cast<double>(match.distance)) / 64.0) * 100.0;
#else
                // Cosine distance over L2-normalized vectors: 0 = identical, 2 = opposite.
                double accuracy = (1.0 - static_cast<double>(match.distance)) * 100.0;
                if (accuracy < 0.0) accuracy = 0.0;
#endif

                out << "{\"id\":" << match.id
                    << ",\"path\":\"" << escapeJSONString(idToPath[match.id])
                    << "\",\"category\":\"" << escapeJSONString(idToCategory[match.id])
                    << "\",\"distance\":" << match.distance
                    << ",\"accuracy\":" << accuracy << "}";
            }
            out << "]}";
            std::cout << out.str() << "\n" << std::flush;
        }
    }
    return 0;
}
