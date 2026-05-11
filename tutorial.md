# How to Start the Project

This is a reverse image search engine: a C++ backend (CNN feature extractor + HNSW graph) wrapped by a Node.js web UI. Images are embedded into 576-d vectors by **MobileNetV3-Small** running on **ONNX Runtime**, and the **HNSW** index searches by cosine similarity.

Setup takes ~5 minutes on a decent connection. The PyTorch download is the slow part (~200 MB).

---

## 1. Install system prerequisites

You need a **C++17 compiler**, **CMake ≥ 3.10**, **Node.js ≥ 18**, **Python ≥ 3.10**, plus `curl`. Pick the section for your OS:

<details>
<summary><strong>Linux / WSL (Debian, Ubuntu)</strong></summary>

```bash
sudo apt update
sudo apt install -y build-essential cmake nodejs npm python3 python3-venv python3-pip curl
```

`python3-venv` and `python3-pip` are critical — without them, `python3 -m venv` creates a broken venv with no `pip` inside.

</details>

<details>
<summary><strong>Linux (Fedora, RHEL)</strong></summary>

```bash
sudo dnf install -y gcc-c++ cmake nodejs npm python3 python3-pip curl
```

</details>

<details>
<summary><strong>macOS</strong></summary>

Install [Homebrew](https://brew.sh/) first if you don't have it, then:

```bash
xcode-select --install   # C++ toolchain (if not already installed)
brew install cmake node python@3.12
```

`curl` is preinstalled on macOS.

</details>

<details>
<summary><strong>Windows (native, no WSL)</strong></summary>

Install each tool from its official installer:

- **Visual Studio Build Tools 2022** with the "Desktop development with C++" workload — provides the MSVC compiler.
- **CMake** — https://cmake.org/download/ (add to PATH during install).
- **Node.js LTS** — https://nodejs.org/ (the installer adds it to PATH automatically).
- **Python 3.10+** — https://www.python.org/downloads/ (check "Add Python to PATH" during install).
- **Git for Windows** — https://git-scm.com/download/win (provides `bash`, `curl`, and `tar` via Git Bash).

Run all subsequent commands inside **Git Bash** so the shell snippets below work identically. PowerShell works too but the `curl | tar` pipe needs adjustment — see notes in the ONNX Runtime step.

</details>

<details>
<summary><strong>Windows (recommended: WSL)</strong></summary>

Install WSL with Ubuntu, then follow the Linux/WSL section above. This is the smoothest path on Windows.

```powershell
wsl --install -d Ubuntu
```

</details>

Verify everything is on PATH:

```bash
g++ --version    # or cl on native Windows
cmake --version
node --version
npm --version
python3 --version
curl --version
```

---

## 2. Download ONNX Runtime

The C++ engine links against a prebuilt ONNX Runtime shared library. From the project root:

<details open>
<summary><strong>Linux x64 / WSL</strong></summary>

```bash
mkdir -p third_party
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz \
  | tar -xz -C third_party/
mv third_party/onnxruntime-linux-x64-1.20.0 third_party/onnxruntime
```

</details>

<details>
<summary><strong>macOS (Apple Silicon — M1/M2/M3)</strong></summary>

```bash
mkdir -p third_party
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-arm64-1.20.0.tgz \
  | tar -xz -C third_party/
mv third_party/onnxruntime-osx-arm64-1.20.0 third_party/onnxruntime
```

</details>

<details>
<summary><strong>macOS (Intel)</strong></summary>

```bash
mkdir -p third_party
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-x86_64-1.20.0.tgz \
  | tar -xz -C third_party/
mv third_party/onnxruntime-osx-x86_64-1.20.0 third_party/onnxruntime
```

</details>

<details>
<summary><strong>Windows x64</strong></summary>

In **Git Bash**:

```bash
mkdir -p third_party
curl -L -o /tmp/ort.zip \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-win-x64-1.20.0.zip
unzip /tmp/ort.zip -d third_party/
mv third_party/onnxruntime-win-x64-1.20.0 third_party/onnxruntime
```

On Windows the binary will need `onnxruntime.dll` (in `third_party/onnxruntime/lib/`) discoverable at runtime. The simplest fix is to copy it next to the built `ReverseImageSearch.exe` after step 4.

</details>

You should now have headers at `third_party/onnxruntime/include/` and the shared library at `third_party/onnxruntime/lib/`.

---

## 3. Export the CNN model

A one-time Python script downloads MobileNetV3-Small from torchvision and writes a self-contained ONNX file.

```bash
python3 -m venv .export_venv

# Linux / macOS
.export_venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.export_venv/bin/pip install onnx onnxscript
.export_venv/bin/python scripts/export_model.py
```

<details>
<summary><strong>Windows (Git Bash) — different venv path</strong></summary>

On Windows, the venv puts executables under `Scripts/` instead of `bin/`:

```bash
python -m venv .export_venv
.export_venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.export_venv/Scripts/pip install onnx onnxscript
.export_venv/Scripts/python scripts/export_model.py
```

</details>

This produces `models/mobilenetv3_small.onnx` (≈3.7 MB). The script strips the classifier and exposes pooled 576-d features.

You can delete `.export_venv/` after this step if disk space matters — it's only needed when re-exporting the model.

---

## 4. Build the C++ engine

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

This produces the `ReverseImageSearch` executable inside `build/` (`.exe` on Windows). Build flags `-O3 -march=native` are set in `CMakeLists.txt`.

**Windows extra step:** copy `onnxruntime.dll` next to the executable so it's found at runtime:

```bash
cp third_party/onnxruntime/lib/onnxruntime.dll build/Release/   # adjust path if CMake put the .exe elsewhere
```

### Backend selection

The HNSW index is templated and supports two backends:

- **CNN embeddings** (default) — MobileNetV3-Small features + cosine distance. Requires steps 2 and 3.
- **pHash** (legacy) — 64-bit DCT pHash + Hamming distance. Pure C++, no ONNX dependency.

To compile the pHash backend, skip steps 2 and 3 and pass `-DUSE_PHASH=ON`:

```bash
cmake -DUSE_PHASH=ON ..
cmake --build .
```

The CMake status line prints the active backend.

The binary is a **long-running process** driven by stdin commands:

- `LOAD <dataset_path>` — recursively scans the folder, embeds every image, builds one HNSW index per subfolder (treated as a category). Prints `{"status":"ready","count":N,"graph":{...}}`.
- `SEARCH <query_image> [category]` — embeds the query, returns the top 12 nearest neighbors. Prints `{"results":[...]}`.

You normally don't run it directly — the Node server spawns it once and pipes commands.

---

## 5. Install web dependencies

```bash
cd web
npm install
```

Dependencies: `express`, `multer`, `cors` (see `web/package.json`).

---

## 6. Run the server

```bash
npm start
```

The server picks a free port automatically and prints something like:

```
Server running at http://localhost:43289
```

Open that URL, paste your dataset folder path, click **Load Dataset** (the CNN embedding pass runs once — expect ~10–30 ms per image), then upload a query image and search.

---

## Project layout

- `main.cpp` — stdin REPL loop, owns the per-category HNSW indexes
- `HNSW.cpp` / `HNSW.h` — templated HNSW index, supports both Hamming and cosine distance
- `ImageProcessor.cpp` / `ImageProcessor.h` — pHash (DCT) and CNN (ONNX Runtime) feature extractors
- `CMakeLists.txt` — build config; `-DUSE_PHASH=ON` selects the legacy backend
- `scripts/export_model.py` — one-time PyTorch → ONNX export
- `models/` — exported `.onnx` file (gitignored, regenerated by the script)
- `third_party/` — `stb_image` (decode/resize) + ONNX Runtime (gitignored)
- `web/` — Node/Express server and frontend in `web/public/`

---

## Troubleshooting

- **`.export_venv/bin/pip: No such file or directory` (Linux)** — `python3-venv` and/or `python3-pip` aren't installed. Run `sudo apt install -y python3-venv python3-pip`, delete the half-broken venv with `rm -rf .export_venv`, and re-run step 3.
- **`ModuleNotFoundError: No module named 'torch'`** — the venv install in step 3 was skipped or failed silently. Re-run step 3 and watch for errors during the `pip install` lines.
- **`cmake: command not found`** — install CMake (see step 1) and open a new terminal so PATH refreshes.
- **`Failed to initialize ONNX model at ...`** — step 3 was skipped, or the file `models/mobilenetv3_small.onnx` is missing. The binary looks for the model at `<exe_dir>/../models/mobilenetv3_small.onnx`.
- **`error while loading shared libraries: libonnxruntime.so.1` (Linux)** — step 2 was skipped or the directory was moved. Re-run CMake from a clean `build/` so the RPATH is re-baked.
- **`The code execution cannot proceed because onnxruntime.dll was not found` (Windows)** — copy `onnxruntime.dll` from `third_party/onnxruntime/lib/` next to your `ReverseImageSearch.exe`.
- **Server can't find the executable** — confirm the binary is at `build/ReverseImageSearch` (or `build/Release/ReverseImageSearch.exe` on Windows) and rebuild if needed.
- **Large dataset feels slow to load** — CNN inference is ~10–30 ms per image on CPU. A 1000-image dataset takes 10–30 seconds. Searches are fast afterwards.
- **Sketch matching still weak** — try raising `efSearch` in `main.cpp` (currently 50), or swap MobileNetV3-Small for CLIP ViT-B/32 in `scripts/export_model.py` and re-run it. CLIP is dramatically better for cross-modal (sketch↔photo) matching.
