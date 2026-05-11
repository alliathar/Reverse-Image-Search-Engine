#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <sstream>
#include "HNSW.h"
#include "ImageProcessor.h"

namespace fs = std::filesystem;

// ----- Backend selection -----
// Define USE_PHASH at compile time (via CMake -DUSE_PHASH=ON) to use the legacy
// 64-bit DCT pHash with Hamming distance. Default is the CNN embedding backend.
#ifdef USE_PHASH
    using HashT = uint64_t;
    using IndexT = PHashHNSW;
    static HashT extractHash(const std::string& path) {
        return ImageProcessor::generatePHash(path);
    }
#else
    using HashT = std::vector<float>;
    using IndexT = EmbeddingHNSW;
    static HashT extractHash(const std::string& path) {
        return ImageProcessor::generateEmbedding(path);
    }
#endif

int main(int /*argc*/, char* argv[]) {
#ifndef USE_PHASH
    // Resolve the model path relative to the executable.
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
            std::cout << "{\"status\":\"ready\",\"count\":" << nextId << ",\"graph\":" << graphJSON << "}\n" << std::flush;

        } else if (words[0] == "SEARCH") {
            if (words.size() < 2) {
                std::cout << "{\"error\":\"Missing query path\"}\n" << std::flush;
                continue;
            }
            std::string queryPath = words[1];
            std::string targetCategory = (words.size() > 2) ? words.back() : "all";

            HashT queryHash;
            try {
                queryHash = extractHash(queryPath);
            } catch (...) {
                std::cout << "{\"error\":\"Could not read query image\"}\n" << std::flush;
                continue;
            }

            struct Match {
                uint64_t id;
                float distance;
                bool operator<(const Match& other) const { return distance < other.distance; }
            };
            std::vector<Match> allMatches;

            auto runOnIndex = [&](IndexT& idx) {
                auto results = idx.search(queryHash, 12, 50);
                for (uint64_t resId : results) {
#ifdef USE_PHASH
                    HammingDistance dfn;
#else
                    CosineDistance dfn;
#endif
                    allMatches.push_back({resId, dfn(queryHash, idx.getEmbedding(resId))});
                }
            };

            if (targetCategory == "all" || targetCategory == "") {
                for (auto& pair : categoryIndexes) runOnIndex(*pair.second);
            } else {
                auto it = categoryIndexes.find(targetCategory);
                if (it == categoryIndexes.end()) {
                    std::cout << "{\"error\":\"Category not found in dataset\"}\n" << std::flush;
                    continue;
                }
                runOnIndex(*it->second);
            }

            std::sort(allMatches.begin(), allMatches.end());
            if (allMatches.size() > 12) allMatches.resize(12);

            std::ostringstream out;
            out << "{\"results\":[";
            bool first = true;
            for (const auto& match : allMatches) {
                if (!first) out << ",";
                first = false;

                // Convert raw distance to a 0-100 "accuracy" score depending on the backend.
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
