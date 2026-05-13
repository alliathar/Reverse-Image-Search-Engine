cmake_minimum_required(VERSION 3.10)
project(ReverseImageSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

# Backend selection: -DUSE_PHASH=ON to compile the legacy 64-bit DCT pHash backend
# (no ONNX Runtime dependency). Default is the CNN embedding backend.
option(USE_PHASH "Use the legacy pHash backend instead of CNN embeddings" OFF)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

add_executable(ReverseImageSearch
    main.cpp
    HNSW.cpp
    ImageProcessor.cpp
)

target_include_directories(ReverseImageSearch PRIVATE third_party)

if(USE_PHASH)
    target_compile_definitions(ReverseImageSearch PRIVATE USE_PHASH)
    message(STATUS "ReverseImageSearch backend: pHash (Hamming distance)")
else()
    target_include_directories(ReverseImageSearch PRIVATE ${ONNXRUNTIME_DIR}/include)
    target_link_libraries(ReverseImageSearch PRIVATE
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    )
    set_target_properties(ReverseImageSearch PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    )
    message(STATUS "ReverseImageSearch backend: CNN embeddings (cosine distance)")
endif()
