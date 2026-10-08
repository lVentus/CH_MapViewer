# CH_MapViewer

CH_MapViewer is a Linux-first OpenGL renderer for contraction-hierarchy-based road-network visualization. Windows is a secondary target.

The current repository establishes the project boundaries, CH/range loading, a CPU reference unfolder, OpenGL infrastructure, Dear ImGui integration, and the runtime-selectable unfolding-strategy interface. GPU unfolding algorithms will be implemented behind that interface and benchmarked against the CPU reference before the road renderer and streaming layer are expanded.

## Current module boundaries

- `core`: application lifecycle and native window ownership
- `data/ch`: CH graph, edge ranges, and file loading
- `reference`: correctness-first CPU algorithms
- `gpu`: OpenGL resources and compute infrastructure
- `gpu/unfolding`: runtime-selectable unfolding strategies
- `renderer`: drawing code only
- `ui`: Dear ImGui tools and runtime controls

Streaming, spatial indexing, and benchmark modules will be added when their first concrete implementation is introduced rather than as empty placeholders.

## Planned unfolding strategies

The renderer uses a stable `IUnfoldingStrategy` interface. Concrete strategies are intended to be switchable at runtime and benchmarked under identical inputs.

Planned candidates include:

- per-thread iterative DFS
- level-synchronous/frontier expansion
- persistent global work queue
- workgroup-local queue variants
- subgroup-cooperative traversal
- workload binning and task splitting
- flattened hierarchy / frontier-query layouts

Traversal strategy and output management are kept separate. Atomic append and prefix-sum/scatter can therefore be tested with more than one traversal method.

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

The first full viewer configuration also needs Internet access because CMake fetches GLFW, GLAD and Dear ImGui from their upstream repositories.

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

Subdirectories are scanned recursively. Command-line graph/range paths are still supported when needed:

```bash
./build-linux/CH_MapViewer /path/to/bw.sch /path/to/bw.sch.ranges
```

For CI or a non-graphics/core-only build, the script also forwards the existing CMake options through environment variables:

```bash
CHMV_BUILD_APP=OFF ./build.sh release
CHMV_BUILD_TESTS=OFF ./build.sh release
```

If you prefer to configure manually on Ubuntu/Debian, the equivalent full dependency set is:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
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

The text loader currently accepts the provided CH format:

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

`-1 -1` ranges are retained in the graph representation but reported as non-drawable.

## Tests

The data/reference layer can be built and tested without fetching graphics dependencies:

```bash
cmake -S . -B build-tests -DCHMV_BUILD_APP=OFF -DCHMV_BUILD_TESTS=ON
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```

## Immediate next milestone

1. Add benchmark input selection and reproducible root-edge sets.
2. Implement per-thread iterative DFS as the first GPU baseline.
3. Implement frontier expansion and persistent-queue variants.
4. Compare correctness against `CPUReferenceUnfolder` and record GPU time, output size, temporary memory, and timing variance.
5. Only after selecting a suitable GPU traversal, connect range filtering and the first road-rendering path.
