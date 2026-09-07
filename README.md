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

Install a compiler, CMake, Ninja, OpenGL development files, and GLFW's X11 dependencies. On Ubuntu/Debian a typical setup is:

```bash
sudo apt install build-essential cmake ninja-build libgl1-mesa-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

Configure and build:

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

CMake fetches GLFW, GLAD, and Dear ImGui during configuration.

Run the viewer and select a dataset from the Dear ImGui Dataset panel. During development, put uncompressed graph/range pairs under `assets/data/`, for example:

```text
assets/data/bw/bw.sch
assets/data/bw/bw.sch.ranges
```

Subdirectories are scanned recursively. Command-line graph/range paths are still supported when needed:

```bash
./build/CH_MapViewer /path/to/bw.sch /path/to/bw.sch.ranges
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
