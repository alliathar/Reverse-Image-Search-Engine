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

extern std::string escapeJSONString(const std::string& input);

int main(int /*argc*/, char* argv[]) {
    // Resolve the model path relative to the executable so the binary works
    // regardless of which directory it's spawned from.
    fs::path exeDir = fs::weakly_canonical(fs::path(argv[0])).parent_path();
    fs::path modelPath = exeDir / ".." / "models" / "mobilenetv3_small.onnx";
    try {
        ImageProcessor::initialize(modelPath.string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX model at " << modelPath << ": " << e.what() << std::endl;
        return 1;
    }

    std::unordered_map<uint64_t, std::string> idToPath;
    std::unordered_map<uint64_t, std::string> idToCategory;
    std::unordered_map<std::string, std::shared_ptr<HNSWIndex>> categoryIndexes;
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
                        Embedding emb = ImageProcessor::generateEmbedding(pathStr);
                        idToPath[nextId] = pathStr;
                        idToCategory[nextId] = category;

                        if (categoryIndexes.find(category) == categoryIndexes.end()) {
                            categoryIndexes[category] = std::make_shared<HNSWIndex>(16, 32, 100);
                        }
                        categoryIndexes[category]->insert(nextId, std::move(emb));
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

            Embedding queryEmb;
            try {
                queryEmb = ImageProcessor::generateEmbedding(queryPath);
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

            if (targetCategory == "all" || targetCategory == "") {
                for (auto& pair : categoryIndexes) {
                    auto results = pair.second->search(queryEmb, 12, 50);
                    for (uint64_t resId : results) {
                        allMatches.push_back({resId, computeDistance(queryEmb, pair.second->getEmbedding(resId))});
                    }
                }
            } else {
                auto it = categoryIndexes.find(targetCategory);
                if (it == categoryIndexes.end()) {
                    std::cout << "{\"error\":\"Category not found in dataset\"}\n" << std::flush;
                    continue;
                }
                auto results = it->second->search(queryEmb, 12, 50);
                for (uint64_t resId : results) {
                    allMatches.push_back({resId, computeDistance(queryEmb, it->second->getEmbedding(resId))});
                }
            }

            std::sort(allMatches.begin(), allMatches.end());
            if (allMatches.size() > 12) allMatches.resize(12);

            std::ostringstream out;
            out << "{\"results\":[";
            bool first = true;
            for (const auto& match : allMatches) {
                if (!first) out << ",";
                first = false;

                // Cosine similarity in [-1, 1] mapped to a 0-100 score.
                double similarity = (1.0 - static_cast<double>(match.distance)) * 100.0;
                if (similarity < 0.0) similarity = 0.0;

                out << "{\"id\":" << match.id
                    << ",\"path\":\"" << escapeJSONString(idToPath[match.id])
                    << "\",\"category\":\"" << escapeJSONString(idToCategory[match.id])
                    << "\",\"distance\":" << match.distance
                    << ",\"accuracy\":" << similarity << "}";
            }
            out << "]}";
            std::cout << out.str() << "\n" << std::flush;
        }
    }
    return 0;
}
