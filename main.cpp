#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <sstream>
#include <cmath>
#include "HNSW.h"
#include "ImageProcessor.h"
#include <bitset>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#ifdef _WIN32
#include <direct.h>
#define MKDIR_P(path) _mkdir(path)
#else
#include <sys/types.h>
#define MKDIR_P(path) mkdir(path, 0755)
#endif

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

void scanDirectory(const std::string& currentPath, const std::string& category, 
                   std::unordered_map<uint64_t, std::string>& idToPath,
                   std::unordered_map<uint64_t, std::string>& idToCategory,
                   std::unordered_map<std::string, std::shared_ptr<IndexT>>& categoryIndexes,
#ifndef USE_PHASH
                   std::unordered_map<uint64_t, uint64_t>& idToPHash,
#endif
                   uint64_t& nextId) {
    DIR* dir = opendir(currentPath.c_str());
    if (!dir) return;
    
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        std::string filename = ent->d_name;
        if (filename == "." || filename == "..") continue;
        
        std::string fullPath = currentPath + "/" + filename;
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                scanDirectory(fullPath, filename, idToPath, idToCategory, categoryIndexes, 
#ifndef USE_PHASH
                              idToPHash, 
#endif
                              nextId);
            } else if (S_ISREG(st.st_mode)) {
                try {
                    HashT hash = extractHash(fullPath);
                    idToPath[nextId] = fullPath;
                    idToCategory[nextId] = category;
                    
                    if (categoryIndexes.find(category) == categoryIndexes.end()) {
                        categoryIndexes[category] = std::make_shared<IndexT>(16, 32, 100);
                    }
                    categoryIndexes[category]->insert(nextId, std::move(hash));
#ifndef USE_PHASH
                    idToPHash[nextId] = ImageProcessor::generatePHash(fullPath);
#endif
                    nextId++;
                } catch (...) {
                    // Skip non-images
                }
            }
        }
    }
    closedir(dir);
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

// FNV-1a hash of a string, returned as a 16-char hex string.
static std::string hashPath(const std::string& path) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : path) {
        h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

// Write a length-prefixed string to a binary stream.
static void writeString(std::ostream& os, const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    os.write(reinterpret_cast<const char*>(&len), sizeof(len));
    os.write(s.data(), len);
}

// Read a length-prefixed string from a binary stream.
static std::string readString(std::istream& is) {
    uint32_t len;
    is.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    is.read(&s[0], len);
    return s;
}

// Save the entire application state (all maps + all HNSW indexes) to a single cache file.
static bool saveCache(const std::string& cachePath,
                      const std::unordered_map<uint64_t, std::string>& idToPath,
                      const std::unordered_map<uint64_t, std::string>& idToCategory,
                      const std::unordered_map<std::string, std::shared_ptr<IndexT>>& categoryIndexes,
#ifndef USE_PHASH
                      const std::unordered_map<uint64_t, uint64_t>& idToPHash,
#endif
                      uint64_t nextId) {
    std::ofstream ofs(cachePath, std::ios::binary);
    if (!ofs) return false;

    // Magic for the outer wrapper
    ofs.write("RCACHE", 6);
    uint32_t cacheVersion = 1;
    ofs.write(reinterpret_cast<const char*>(&cacheVersion), sizeof(cacheVersion));

    // nextId
    ofs.write(reinterpret_cast<const char*>(&nextId), sizeof(nextId));

    // idToPath
    uint64_t pathCount = idToPath.size();
    ofs.write(reinterpret_cast<const char*>(&pathCount), sizeof(pathCount));
    for (const auto& p : idToPath) {
        ofs.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
        writeString(ofs, p.second);
    }

    // idToCategory
    uint64_t catCount = idToCategory.size();
    ofs.write(reinterpret_cast<const char*>(&catCount), sizeof(catCount));
    for (const auto& p : idToCategory) {
        ofs.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
        writeString(ofs, p.second);
    }

#ifndef USE_PHASH
    // idToPHash
    uint64_t phCount = idToPHash.size();
    ofs.write(reinterpret_cast<const char*>(&phCount), sizeof(phCount));
    for (const auto& p : idToPHash) {
        ofs.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
        ofs.write(reinterpret_cast<const char*>(&p.second), sizeof(p.second));
    }
#endif

    // Category indexes
    uint32_t idxCount = static_cast<uint32_t>(categoryIndexes.size());
    ofs.write(reinterpret_cast<const char*>(&idxCount), sizeof(idxCount));
    for (const auto& p : categoryIndexes) {
        writeString(ofs, p.first);
        p.second->save(ofs);
    }

    return ofs.good();
}

// Load the entire application state from a cache file.
static bool loadCache(const std::string& cachePath,
                      std::unordered_map<uint64_t, std::string>& idToPath,
                      std::unordered_map<uint64_t, std::string>& idToCategory,
                      std::unordered_map<std::string, std::shared_ptr<IndexT>>& categoryIndexes,
#ifndef USE_PHASH
                      std::unordered_map<uint64_t, uint64_t>& idToPHash,
#endif
                      uint64_t& nextId) {
    std::ifstream ifs(cachePath, std::ios::binary);
    if (!ifs) return false;

    // Check magic
    char magic[6];
    ifs.read(magic, 6);
    if (std::string(magic, 6) != "RCACHE") return false;

    uint32_t cacheVersion;
    ifs.read(reinterpret_cast<char*>(&cacheVersion), sizeof(cacheVersion));
    if (cacheVersion != 1) return false;

    // nextId
    ifs.read(reinterpret_cast<char*>(&nextId), sizeof(nextId));

    // idToPath
    uint64_t pathCount;
    ifs.read(reinterpret_cast<char*>(&pathCount), sizeof(pathCount));
    for (uint64_t i = 0; i < pathCount; ++i) {
        uint64_t id;
        ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
        idToPath[id] = readString(ifs);
    }

    // idToCategory
    uint64_t catCount;
    ifs.read(reinterpret_cast<char*>(&catCount), sizeof(catCount));
    for (uint64_t i = 0; i < catCount; ++i) {
        uint64_t id;
        ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
        idToCategory[id] = readString(ifs);
    }

#ifndef USE_PHASH
    // idToPHash
    uint64_t phCount;
    ifs.read(reinterpret_cast<char*>(&phCount), sizeof(phCount));
    for (uint64_t i = 0; i < phCount; ++i) {
        uint64_t id, ph;
        ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
        ifs.read(reinterpret_cast<char*>(&ph), sizeof(ph));
        idToPHash[id] = ph;
    }
#endif

    // Category indexes
    uint32_t idxCount;
    ifs.read(reinterpret_cast<char*>(&idxCount), sizeof(idxCount));
    for (uint32_t i = 0; i < idxCount; ++i) {
        std::string name = readString(ifs);
        auto idx = std::make_shared<IndexT>(16, 32, 100);
        idx->load(ifs);
        categoryIndexes[name] = idx;
    }

    return ifs.good();
}

int main(int /*argc*/, char* argv[]) {
#ifndef USE_PHASH
    std::string exePathStr(argv[0]);
    size_t lastSlash = exePathStr.find_last_of("\\/");
    std::string exeDir = (lastSlash == std::string::npos) ? "." : exePathStr.substr(0, lastSlash);
    std::string modelPath = exeDir + "/../models/mobilenetv3_small.onnx";
    try {
        ImageProcessor::initialize(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX model at " << modelPath << ": " << e.what() << std::endl;
        return 1;
    }
#endif

    std::unordered_map<uint64_t, std::string> idToPath;
    std::unordered_map<uint64_t, std::string> idToCategory;
    std::unordered_map<std::string, std::shared_ptr<IndexT>> categoryIndexes;
#ifndef USE_PHASH
    std::unordered_map<uint64_t, uint64_t> idToPHash; // For UI feature analysis
#endif
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
#ifndef USE_PHASH
            idToPHash.clear();
#endif
            nextId = 0;

            std::string datasetPath;
            for (size_t i = 1; i < words.size(); i++) {
                if (i > 1) datasetPath += " ";
                datasetPath += words[i];
            }

            // Resolve cache directory relative to the executable.
            std::string exePath2(argv[0]);
            size_t sl2 = exePath2.find_last_of("\\/");
            std::string ed2 = (sl2 == std::string::npos) ? "." : exePath2.substr(0, sl2);
            std::string cacheDir = ed2 + "/../cache";
            MKDIR_P(cacheDir.c_str());

            std::string cacheFile = cacheDir + "/" + hashPath(datasetPath) + ".bin";
            bool cacheHit = false;

            // Try loading from cache first.
            try {
                cacheHit = loadCache(cacheFile, idToPath, idToCategory, categoryIndexes,
#ifndef USE_PHASH
                                     idToPHash,
#endif
                                     nextId);
            } catch (...) {
                cacheHit = false;
            }

            if (!cacheHit) {
                // Cache miss — scan the dataset from scratch.
                idToPath.clear();
                idToCategory.clear();
                categoryIndexes.clear();
#ifndef USE_PHASH
                idToPHash.clear();
#endif
                nextId = 0;

                try {
                    scanDirectory(datasetPath, "default", idToPath, idToCategory, categoryIndexes, 
#ifndef USE_PHASH
                                  idToPHash, 
#endif
                                  nextId);
                } catch (const std::exception& e) {
                    std::cout << "{\"error\":\"" << escapeJSONString(e.what()) << "\"}\n" << std::flush;
                    continue;
                }

                // Save to cache for next time.
                saveCache(cacheFile, idToPath, idToCategory, categoryIndexes,
#ifndef USE_PHASH
                          idToPHash,
#endif
                          nextId);
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
            out << "{\"status\":\"ready\",\"count\":" << nextId
                << ",\"cached\":" << (cacheHit ? "true" : "false")
                << ",\"categories\":[";
            for (size_t i = 0; i < categoryNames.size(); ++i) {
                if (i > 0) out << ",";
                out << "\"" << escapeJSONString(categoryNames[i]) << "\"";
            }
            out << "],\"graph\":" << graphJSON << "}";
            std::cout << out.str() << "\n" << std::flush;

        } else if (words[0] == "SEARCH") {
            // Protocol: SEARCH <mode> <include_csv> <exclude_csv> <path1> [<path2>...]
            //   mode:        "normal" | "negative" | "multi"
            //   include_csv: comma-separated category names to keep, or "_" for all
            //   exclude_csv: comma-separated category names to drop, or "_" for none
            //   Category names can't contain commas or spaces (folder names from the dataset).
            if (words.size() < 5) {
                std::cout << "{\"error\":\"Usage: SEARCH <mode> <include> <exclude> <path>...\"}\n" << std::flush;
                continue;
            }
            std::string mode = words[1];
            std::string includeArg = words[2];
            std::string excludeArg = words[3];
            std::vector<std::string> queryPaths(words.begin() + 4, words.end());

            auto splitCSV = [](const std::string& s) {
                std::vector<std::string> out;
                if (s == "_" || s.empty()) return out;
                size_t start = 0;
                for (size_t i = 0; i <= s.size(); ++i) {
                    if (i == s.size() || s[i] == ',') {
                        if (i > start) out.emplace_back(s.substr(start, i - start));
                        start = i + 1;
                    }
                }
                return out;
            };
            std::vector<std::string> includeList = splitCSV(includeArg);
            std::vector<std::string> excludeList = splitCSV(excludeArg);
            std::unordered_set<std::string> excludeSet(excludeList.begin(), excludeList.end());

            // Compute / combine query hashes.
            HashT queryHash;
#ifndef USE_PHASH
            uint64_t queryPHashUI = 0;
#endif
            try {
                if (mode == "multi" && queryPaths.size() > 1) {
                    std::vector<HashT> hashes;
                    hashes.reserve(queryPaths.size());
#ifndef USE_PHASH
                    std::vector<uint64_t> pHashes;
                    pHashes.reserve(queryPaths.size());
#endif
                    for (const auto& p : queryPaths) {
                        hashes.push_back(extractHash(p));
#ifndef USE_PHASH
                        pHashes.push_back(ImageProcessor::generatePHash(p));
#endif
                    }
                    queryHash = combineHashes(hashes);
#ifndef USE_PHASH
                    // Combine pHashes for UI
                    std::vector<int> bitCounts(64, 0);
                    for (uint64_t h : pHashes) {
                        for (int i = 0; i < 64; ++i) {
                            if (h & (1ULL << i)) bitCounts[i]++;
                        }
                    }
                    int threshold = static_cast<int>(pHashes.size() + 1) / 2;
                    for (int i = 0; i < 64; ++i) {
                        if (bitCounts[i] >= threshold) queryPHashUI |= (1ULL << i);
                    }
#endif
                } else {
                    queryHash = extractHash(queryPaths[0]);
#ifndef USE_PHASH
                    queryPHashUI = ImageProcessor::generatePHash(queryPaths[0]);
#endif
                }
            } catch (...) {
                std::cout << "{\"error\":\"Could not read query image\"}\n" << std::flush;
                continue;
            }

            struct Match {
                uint64_t id;
                float distance;
                uint64_t hash; // Always store pHash for the UI
            };
            std::vector<Match> allMatches;
            DistanceFn dfn;

            // Resolve which indexes to scan.
            // Base set: explicit include list if non-empty, otherwise all categories.
            // Then filter out any category in the exclude set.
            std::vector<std::shared_ptr<IndexT>> targetIndexes;
            if (includeList.empty()) {
                for (auto& p : categoryIndexes) {
                    if (excludeSet.count(p.first)) continue;
                    targetIndexes.push_back(p.second);
                }
            } else {
                for (const auto& name : includeList) {
                    if (excludeSet.count(name)) continue;
                    auto it = categoryIndexes.find(name);
                    if (it != categoryIndexes.end()) targetIndexes.push_back(it->second);
                }
            }
            if (targetIndexes.empty()) {
                std::cout << "{\"error\":\"No categories left after include/exclude filtering\"}\n" << std::flush;
                continue;
            }

            if (mode == "negative") {
                // Reverse-greedy HNSW traversal. Finds a local maximum of distance
                // rather than the true global furthest, but in practice that's fine —
                // the result is still a strongly-dissimilar image, and the search is
                // O(log N) instead of O(N).
                for (auto& idx : targetIndexes) {
                    auto results = idx->searchFurthest(queryHash, 12, 50);
                    for (uint64_t resId : results) {
#ifdef USE_PHASH
                        allMatches.push_back({resId, dfn(queryHash, idx->getEmbedding(resId)), idx->getEmbedding(resId)});
#else
                        allMatches.push_back({resId, dfn(queryHash, idx->getEmbedding(resId)), idToPHash[resId]});
#endif
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
#ifdef USE_PHASH
                        allMatches.push_back({resId, dfn(queryHash, idx->getEmbedding(resId)), idx->getEmbedding(resId)});
#else
                        allMatches.push_back({resId, dfn(queryHash, idx->getEmbedding(resId)), idToPHash[resId]});
#endif
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
                    << ",\"accuracy\":" << accuracy
                    << ",\"hash\":\"" << std::bitset<64>(match.hash).to_string() << "\"}";
            }
            out << "]";
#ifdef USE_PHASH
            out << ",\"queryHash\":\"" << std::bitset<64>(queryHash).to_string() << "\"";
#else
            out << ",\"queryHash\":\"" << std::bitset<64>(queryPHashUI).to_string() << "\"";
#endif
            out << "}";
            std::cout << out.str() << "\n" << std::flush;
        }
    }
    return 0;
}
