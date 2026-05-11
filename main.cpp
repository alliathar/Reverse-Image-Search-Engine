#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <bitset>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include "HNSW.h"
#include "ImageProcessor.h"

namespace fs = std::filesystem;

// External declaration from HNSW.cpp for escaping logic
extern std::string escapeJSONString(const std::string& input);

int main(int argc, char* argv[]) {
    // Usage: ReverseImageSearch.exe search <query> <dataset_dir> [category]
    if (argc >= 4 && std::string(argv[1]) == "search") {
        std::string queryPath = argv[2];
        std::string datasetPath = argv[3];
        std::string targetCategory = (argc >= 5) ? argv[4] : "all";
        
        std::unordered_map<std::string, std::shared_ptr<HNSWIndex>> categoryIndexes;
        std::unordered_map<uint64_t, std::string> idToPath;
        std::unordered_map<uint64_t, uint64_t> idToHash;
        std::unordered_map<uint64_t, std::string> idToCategory;
        uint64_t nextId = 0;
        
        if (!fs::exists(datasetPath) || !fs::is_directory(datasetPath)) {
            std::cout << "{\"error\": \"Dataset path is invalid or not a directory\"}\n";
            return 1;
        }

        // 1. Hash and Insert target images into respective category graphs
        for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
            if (entry.is_regular_file()) {
                std::string pathStr = entry.path().string();
                std::string category = entry.path().parent_path().filename().string();
                
                try {
                    uint64_t hash = ImageProcessor::generatePHash(pathStr);
                    idToPath[nextId] = pathStr;
                    idToHash[nextId] = hash;
                    idToCategory[nextId] = category;
                    
                    if (categoryIndexes.find(category) == categoryIndexes.end()) {
                        categoryIndexes[category] = std::make_shared<HNSWIndex>(16, 32, 100);
                    }
                    categoryIndexes[category]->insert(nextId, hash);
                    nextId++;
                } catch (...) {
                    // Skip non-images or unreadable files
                }
            }
        }
        
        if (nextId == 0) {
            std::cout << "{\"error\": \"No valid images found in dataset\"}\n";
            return 1;
        }

        // 2. Query
        uint64_t queryHash = 0;
        try {
            queryHash = ImageProcessor::generatePHash(queryPath);
        } catch (...) {
            std::cout << "{\"error\": \"Could not read query image\"}\n";
            return 1;
        }
        
        struct Match {
            uint64_t id;
            uint32_t distance;
            bool operator<(const Match& other) const {
                return distance < other.distance;
            }
        };
        std::vector<Match> allMatches;
        
        // Search requested category, or all categories
        if (targetCategory == "all" || targetCategory == "") {
            for (auto& pair : categoryIndexes) {
                auto results = pair.second->search(queryHash, 12, 50);
                for (uint64_t resId : results) {
                    allMatches.push_back({resId, computeHammingDistance(queryHash, idToHash[resId])});
                }
            }
        } else {
            if (categoryIndexes.find(targetCategory) != categoryIndexes.end()) {
                auto results = categoryIndexes[targetCategory]->search(queryHash, 12, 50);
                for (uint64_t resId : results) {
                    allMatches.push_back({resId, computeHammingDistance(queryHash, idToHash[resId])});
                }
            } else {
                std::cout << "{\"error\": \"Category not found in dataset\"}\n";
                return 1;
            }
        }
        
        std::sort(allMatches.begin(), allMatches.end());
        if (allMatches.size() > 12) {
            allMatches.resize(12);
        }
        
        // 3. Prepare JSON Output
        std::cout << "{\n  \"queryHash\": \"" << std::bitset<64>(queryHash).to_string() << "\",\n  \"results\": [\n";
        bool first = true;
        for (const auto& match : allMatches) {
            if (!first) std::cout << ",\n";
            first = false;
            
            uint64_t resId = match.id;
            uint32_t distance = match.distance;
            double accuracy = ((64.0 - static_cast<double>(distance)) / 64.0) * 100.0;
            
            std::cout << "    {\"id\": " << resId 
                      << ", \"path\": \"" << escapeJSONString(idToPath[resId]) 
                      << "\", \"category\": \"" << escapeJSONString(idToCategory[resId])
                      << "\", \"distance\": " << distance 
                      << ", \"accuracy\": " << accuracy 
                      << ", \"hash\": \"" << std::bitset<64>(idToHash[resId]).to_string() << "\"}";
        }
        std::cout << "\n  ],\n";
        
        // 4. Output graph structure of the category containing the best match
        std::shared_ptr<HNSWIndex> bestGraph = nullptr;
        if (!allMatches.empty()) {
            std::string bestCategory = idToCategory[allMatches[0].id];
            bestGraph = categoryIndexes[bestCategory];
        }

        std::cout << "  \"graph\": {\n";
        if (bestGraph) {
            std::string graphJSON = bestGraph->exportGraphJSON(idToPath);
            if (graphJSON.length() > 2) {
                std::string innerGraph = graphJSON.substr(1, graphJSON.length() - 2);
                std::cout << innerGraph;
            }
        }
        std::cout << "\n  }\n}\n";
        return 0;
    }
    
    std::cout << "Usage: ReverseImageSearch.exe search <query_image> <dataset_directory> [category]\n";
    return 1;
}
