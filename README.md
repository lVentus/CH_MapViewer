# CH_MapViewer

CH_MapViewer is a Linux-first OpenGL renderer for contraction-hierarchy-based road-network visualization. Windows is a secondary target.

A dataset (`.sch` + `.sch.ranges`) is turned once into a self-contained page file (`<name>.chlod`, built in the background and checked against plain DFS). The viewer then has three independent parts:

- **Loading (virtual map):** the pages a view needs follow from the camera alone; they are streamed into a GPU page pool from RAM (when the page file fits) or from disk. All pages can be preloaded into the GPU when they fit.
- **Filter:** finds the records of the current level on the GPU — full scan, birth-ordered (records sorted per page, prefix search), or GPU page culling (the GPU picks the pages itself).
- **Expansion:** decides what is drawn — none (roots only), full BFS, adaptive DFS, adaptive BFS, or dual interval (no traversal: every record decides from its stored intervals).

Filter and expansion are chosen at runtime; every combination draws the same roads, which the viewer can check against a CPU evaluation and benchmark on the GPU.

## Module boundaries

- `core`: application lifecycle and native window ownership
- `data`: dataset catalog, background loader, memory queries
- `streaming/lod`: page file format, builder, reader and page planner
- `gpu`: OpenGL resources and compute infrastructure
- `gpu/lod`: GPU page pool, pass orchestration and the runtime
- `gpu/filtering`, `gpu/unfolding`: the runtime-selectable strategies (`IRangeFilterStrategy`, `IUnfoldingStrategy`); a new strategy is one class plus its shaders, registered in `Application`
- `renderer`: camera, LOD curve and frame setup
- `benchmark`: GPU timers and the zoom sweep
- `ui`: Dear ImGui tools and runtime controls

## Build

### Linux

The recommended Linux entry point is `build.sh`. It detects the distribution, makes sure the required system packages are installed, and then configures and builds the project with CMake + Ninja.

For a Release build:

```bash
./build.sh release
```

For the default Debug build:

```bash
./build.sh
```

The script currently handles the main Linux paths as follows:

- **NixOS / Nix shell:** uses the repository `flake.nix` through `nix develop`.
- **Ubuntu / Debian and derivatives:** installs missing compiler, CMake/Ninja, OpenGL, X11, Wayland, Python/Jinja2 and Git packages with `apt-get`, then builds natively.
- **Fedora / RHEL family:** installs the equivalent packages with `dnf`.
- **Arch family:** installs the equivalent packages with `pacman`.
- **openSUSE / SLES:** installs the equivalent packages with `zypper`.
- On an otherwise unsupported distribution, the script falls back to `flake.nix` when Nix is available.

System packages are only installed when they are missing. If installation requires elevated privileges, the script uses `sudo` (or runs the package manager directly when already root).

The first full viewer configuration also needs Internet access because CMake fetches GLFW, GLAD and Dear ImGui from their upstream repositories. zstd (for the compressed page files) is taken from the system when it is installed (`build.sh` installs it) and otherwise fetched and built as well.

Other useful commands are:

```bash
./build.sh debug
./build.sh rebuild
./build.sh clean
```

The Linux viewer binary is written to:

```text
build-linux/CH_MapViewer
```

Run it with:

```bash
./build-linux/CH_MapViewer
```

During development, put uncompressed graph/range pairs under `assets/data/`, for example:

```text
assets/data/bw/bw.sch
assets/data/bw/bw.sch.ranges
```

Subdirectories are scanned recursively. Command-line graph/range paths (or a page file alone) are still supported when needed:

```bash
./build-linux/CH_MapViewer /path/to/bw.sch /path/to/bw.sch.ranges
./build-linux/CH_MapViewer /path/to/bw.chlod
```

Loading a dataset builds `<name>.chlod` next to it when that file is missing, older than the sources, damaged or of an older format; a page file of the previous version (3, the same data uncompressed) is only compressed, not rebuilt. The page file holds the complete CH as well, zstd-compressed (bw: about 430 MiB instead of 1.2 GB uncompressed), so it also works on its own: a `.chlod` whose `.sch`/`.ranges` are gone is listed and loaded as it is (it can only be rebuilt with them).

Road colors and widths come from `assets/road_styles.cfg`: a few road classes over the edge types of OsmGraphCreator's `all.cfg`. The viewer reloads the file when it is saved.

For CI or a non-graphics/core-only build, the script also forwards the existing CMake options through environment variables:

```bash
CHMV_BUILD_APP=OFF ./build.sh release
CHMV_BUILD_TESTS=OFF ./build.sh release
```

If you prefer to configure manually on Ubuntu/Debian, the equivalent full dependency set is:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config libzstd-dev \
    python3 python3-jinja2 libgl1-mesa-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libwayland-dev wayland-protocols libxkbcommon-dev

cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux --parallel
```

### Windows

Use a recent Visual Studio installation with C++ and CMake support. CH_MapViewer requires a 64-bit build; do not configure it as Win32.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

In VS Code with CMake Tools, select an x64 kit/variant before configuring. If an existing `build` directory was configured as Win32, delete it before reconfiguring.

## Data format

The page file builder reads the provided CH format:

```text
<NodeCount>
<EdgeCount>
<ID> <OSMID> <Lat> <Lon> <Height> <Level>
...
<SrcID> <TrgID> <Weight> <Type> <MaxSpeed> <EdgeIdA> <EdgeIdB>
...
```

and the range file:

```text
<EdgeID> <birthLevel> <deathLevel>
```

`-1 -1` ranges are kept in the CH layer of the page file but never drawn.

The page file (`streaming/lod/LodPageFormat.h`) stores every column in zstd blocks and every page (and its children block) as its own zstd frame, so a single page is still read and decoded on its own. The builder checks the uncompressed file against DFS first and keeps the compressed one only once it decodes to exactly the same bytes.

## Tests

The data layer (page file builder, reader, planner, loader) can be built and tested without fetching graphics dependencies:

```bash
cmake -S . -B build-tests -DCHMV_BUILD_APP=OFF -DCHMV_BUILD_TESTS=ON
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```
