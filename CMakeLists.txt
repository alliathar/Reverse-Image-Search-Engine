cmake_minimum_required(VERSION 3.10)
project(ReverseImageSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimize for speed to test O(1) Hamming distance and O(log N) runtime
set(CMAKE_CXX_FLAGS "-O3 -march=native")

add_executable(ReverseImageSearch 
    main.cpp
    HNSW.cpp
    ImageProcessor.cpp
)

target_include_directories(ReverseImageSearch PRIVATE third_party)
