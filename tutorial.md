# How to Start the Project

This is a reverse image search engine: a C++ backend (HNSW graph + pHash) wrapped by a Node.js web UI.

## Prerequisites

- C++17 compiler (GCC / Clang / MSVC)
- CMake ≥ 3.10
- Node.js ≥ 18 (includes `npm`)

## 1. Build the C++ engine

From the project root ([Reverse-Image-Search-Engine/](./)):

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

This produces the `ReverseImageSearch` executable (`.exe` on Windows) inside [build/](./build/). Build flags `-O3 -march=native` are set in [CMakeLists.txt](./CMakeLists.txt).

Quick sanity check — run the binary directly:

```bash
./ReverseImageSearch search <query_image> <dataset_directory>
```

It prints JSON with the top matches and the HNSW graph (see [main.cpp:15-99](./main.cpp#L15-L99)).

## 2. Install web dependencies

In a new terminal, from the project root:

```bash
cd web
npm install
```

Dependencies: `express`, `multer`, `cors` (see [web/package.json](./web/package.json)).

## 3. Run the server

Still inside [web/](./web):

```bash
npm start
```

The server listens on **http://localhost:3000** ([web/server.js:9](./web/server.js#L9)).

Open the URL in a browser, point the UI at a dataset folder (absolute path), upload a query image, and the matched results and HNSW graph will render.

## Project layout

- [main.cpp](./main.cpp) — CLI entry point, wires up indexing + search
- [HNSW.cpp](./HNSW.cpp) / [HNSW.h](./HNSW.h) — HNSW graph index
- [ImageProcessor.cpp](./ImageProcessor.cpp) / [ImageProcessor.h](./ImageProcessor.h) — pHash generation
- [CMakeLists.txt](./CMakeLists.txt) — build config
- [third_party/](./third_party/) — `stb_image`, `nlohmann/json`
- [web/](./web/) — Node/Express server and frontend in [web/public/](./web/public/)

## Troubleshooting

- **`cmake` not found** — install CMake and re-open the terminal.
- **Server can't find the executable** — confirm the binary is at [build/ReverseImageSearch](./build/ReverseImageSearch) (or wherever [web/server.js](./web/server.js) expects it) and rebuild if needed.
- **Port 3000 in use** — edit the `port` constant in [web/server.js:9](./web/server.js#L9).
- **`{"error": "Dataset path is invalid"}`** — pass an absolute path that exists and contains readable images.
