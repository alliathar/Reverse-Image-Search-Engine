# Setup Tutorial: HNSW Reverse Image Search Engine

Welcome to the HNSW Reverse Image Search Engine project! This guide will walk you through setting up and running the application on any local computer.

The application consists of a high-performance C++ backend utilizing an HNSW (Hierarchical Navigable Small World) graph and pHashes, coupled with a Node.js Express server to provide a rich web interface.

## Prerequisites

To run this project locally, ensure you have the following installed on your system:

1. **C++ Compiler**: A modern C++ compiler that supports C++17 (e.g., GCC, Clang, or MSVC).
2. **CMake**: Version 3.10 or higher.
3. **Node.js**: Version 18.x or higher (comes with `npm`).
4. **Git**: For cloning and version control.

## 1. Backend Setup (C++ Engine)

The backend handles the computationally intensive task of generating pHashes and searching the HNSW graph.

1. **Open a terminal** and navigate to the root directory of the project.
2. **Create a build directory**:
   ```bash
   mkdir build
   cd build
   ```
3. **Generate build files with CMake**:
   ```bash
   cmake ..
   ```
4. **Compile the executable**:
   ```bash
   cmake --build . --config Release
   ```
   *(Note: The `CMakeLists.txt` already applies `-O3 -march=native` flags for high-performance optimization).*
5. You should now see the `ReverseImageSearch` executable (`ReverseImageSearch.exe` on Windows) generated in the `build` or root directory. 

## 2. Frontend Setup (Node.js Server)

The frontend provides an interactive physics-based graph UI for querying images.

1. **Open a new terminal window/tab** and navigate to the `web` folder:
   ```bash
   cd web
   ```
2. **Install the Node.js dependencies** (Express, multer, etc.):
   ```bash
   npm install
   ```

## 3. Running the Application

Once both the backend is compiled and the frontend dependencies are installed, you can start the application!

1. In the `web` directory terminal, **start the Node.js server**:
   ```bash
   npm start
   ```
   *If `npm start` is not configured, use `node server.js`.*

2. **Open your web browser** and navigate to:
   [http://localhost:3000](http://localhost:3000) (Or the port specified in `server.js`).

3. From the UI, you can specify a dataset folder (absolute path) and upload a query image to see the HNSW graph and matched results!

## Project Structure Overview

- **`main.cpp`, `HNSW.cpp`, `ImageProcessor.cpp`**: Core C++ codebase handling pHash generation and nearest neighbor search.
- **`CMakeLists.txt`**: Build configuration for the C++ backend.
- **`web/`**: Node.js backend (`server.js`) and modern web frontend (`public/`).
- **`third_party/`**: External dependencies (like `stb_image` for reading images and `nlohmann/json` for JSON serialization).

Enjoy using the Image Search Engine!
