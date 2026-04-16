#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <bitset>
#include <fstream>
#include <dirent.h>
#include <unordered_map>
#include "HNSW.h"
#include "ImageProcessor.h"

// External declaration from HNSW.cpp for escaping logic
extern std::string escapeJSONString(const std::string& input);

int main(int argc, char* argv[]) {
    // Usage: ReverseImageSearch.exe search <query> <dataset_dir>
    if (argc >= 4 && std::string(argv[1]) == "search") {
        std::string queryPath = argv[2];
        std::string datasetPath = argv[3];
        
        HNSWIndex index(16, 32, 100);
        std::unordered_map<uint64_t, std::string> idToPath;
        std::unordered_map<uint64_t, uint64_t> idToHash;
        uint64_t nextId = 0;
        
        // 1. Hash and Insert target images
        if (!datasetPath.empty() && datasetPath.back() != '/' && datasetPath.back() != '\\') {
            datasetPath += "/";
        }
        
        DIR* dir = opendir(datasetPath.c_str());
        if (dir != nullptr) {
            struct dirent* ent;
            while ((ent = readdir(dir)) != nullptr) {
                std::string fname = ent->d_name;
                if (fname == "." || fname == "..") continue;
                std::string pathStr = datasetPath + fname;
                
                try {
                    uint64_t hash = ImageProcessor::generatePHash(pathStr);
                    idToPath[nextId] = pathStr;
                    idToHash[nextId] = hash;
                    index.insert(nextId, hash);
                    nextId++;
                } catch (...) {
                    // Skip non-images or unreadable files
                }
            }
            closedir(dir);
        } else {
            std::cout << "{\"error\": \"Dataset path is invalid\"}\n";
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
        
        auto results = index.search(queryHash, 12, 50); // Get top 12 matches
        
        // 3. Prepare JSON Output
        std::cout << "{\n  \"results\": [\n";
        bool first = true;
        for (uint64_t resId : results) {
            if (!first) std::cout << ",\n";
            first = false;
            
            uint32_t distance = computeHammingDistance(queryHash, idToHash[resId]);
            double accuracy = ((64.0 - static_cast<double>(distance)) / 64.0) * 100.0;
            
            std::cout << "    {\"id\": " << resId 
                      << ", \"path\": \"" << escapeJSONString(idToPath[resId]) 
                      << "\", \"distance\": " << distance 
                      << ", \"accuracy\": " << accuracy << "}";
        }
        std::cout << "\n  ],\n";
        
        // Print graph structure
        std::string graphJSON = index.exportGraphJSON(idToPath);
        std::cout << "  \"graph\": {\n";
        
        // Substring to remove the outer braces '{ \n ... \n }' from graph JSON
        if (graphJSON.length() > 2) {
            std::string innerGraph = graphJSON.substr(1, graphJSON.length() - 2);
            std::cout << innerGraph;
        }
        
        std::cout << "\n  }\n}\n";
        return 0;
    }
    
    std::cout << "Usage: ReverseImageSearch.exe search <query_image> <dataset_directory>\n";
    return 1;
}
