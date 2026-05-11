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
#include <sstream>

namespace fs = std::filesystem;

// External declaration from HNSW.cpp for escaping logic
extern std::string escapeJSONString(const std::string& input);

int main(int argc, char* argv[]) {
    HNSWIndex index(16, 32, 100);
    std::unordered_map<uint64_t, std::string> idToPath;
    std::unordered_map<uint64_t, std::uint64_t> idToHash;
    std::unordered_map<uint64_t, std::string> idToCategory;
    std::unordered_map<std::string, std::shared_ptr<HNSWIndex>> categoryIndexes;

    uint64_t nextId = 0;

    std::string line;
    while(std::getline(std::cin, line)){
        std::stringstream ss(line);
        std::string word;
        std::vector<std::string> words;
        while(ss >> word){
            words.push_back(word);
        }
        if(words.empty()) continue;
        //LOAD <dataset_path>
        if(words[0] == "LOAD"){
            idToPath.clear();
            idToHash.clear();
            idToCategory.clear();
            categoryIndexes.clear();
            nextId = 0;
            index = HNSWIndex(16, 32, 100);
            std::string datasetPath;

            for (size_t i = 1; i < words.size(); i++) {
                if (i > 1) datasetPath += " ";
                datasetPath += words[i];
            }

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
            std::string graphJSON = "{}";
            if (!categoryIndexes.empty()) {
                graphJSON = categoryIndexes.begin()->second->exportGraphJSON(idToPath);
                graphJSON.erase(std::remove(graphJSON.begin(), graphJSON.end(), '\n'), graphJSON.end());
            }
            std::cout << "{\"status\":\"ready\",\"count\":" << nextId << ",\"graph\":" << graphJSON << "}\n" << std::flush;

        
            //SEARCH <query path> [category]
        }else if(words[0] == "SEARCH"){
            std::string queryPath = words[1];
            std::string targetCategory = (words.size() > 2) ? words.back() : "all";
            uint64_t queryHash = 0;
            try {
                queryHash = ImageProcessor::generatePHash(queryPath);
            } catch (...) {
                std::cout << "{\"error\": \"Could not read query image\"}\n";
                std::cout << std::flush; continue;
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
                    std::cout << std::flush; continue;
                }
            }
            
            std::sort(allMatches.begin(), allMatches.end());
            if (allMatches.size() > 12) {
                allMatches.resize(12);
            }
            
            std::ostringstream out;
            out << "{\"queryHash\":\"" << std::bitset<64>(queryHash).to_string() << "\",\"results\":[";
            bool first = true;
            for (const auto& match : allMatches) {
                if (!first) out << ",";
                first = false;

                uint64_t resId = match.id;
                uint32_t distance = match.distance;
                double accuracy = ((64.0 - static_cast<double>(distance)) / 64.0) * 100.0;

                out << "{\"id\":" << resId
                    << ",\"path\":\"" << escapeJSONString(idToPath[resId])
                    << "\",\"category\":\"" << escapeJSONString(idToCategory[resId])
                    << "\",\"distance\":" << distance
                    << ",\"accuracy\":" << accuracy
                    << ",\"hash\":\"" << std::bitset<64>(idToHash[resId]).to_string() << "\"}";
            }
            out << "]}";
            std::cout << out.str() << "\n" << std::flush; continue;
        }
    }

    return 1;
}
