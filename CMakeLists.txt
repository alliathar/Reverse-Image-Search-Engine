cmake_minimum_required(VERSION 3.10)
project(ReverseImageSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

add_executable(ReverseImageSearch
    main.cpp
    HNSW.cpp
    ImageProcessor.cpp
)

target_include_directories(ReverseImageSearch PRIVATE
    third_party
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(ReverseImageSearch PRIVATE
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

# Embed the ONNX Runtime lib directory in the binary's RPATH so libonnxruntime.so
# is found at runtime without needing LD_LIBRARY_PATH.
set_target_properties(ReverseImageSearch PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)
