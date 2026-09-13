#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-linux"
BUILD_TYPE="Debug"
ACTION="build"

usage() {
    cat <<'USAGE'
Usage:
  ./build.sh
  ./build.sh debug
  ./build.sh release
  ./build.sh rebuild
  ./build.sh clean

Build output:
  build-linux/CH_MapViewer
USAGE
}

case "${1:-}" in
    "")
        ;;
    debug|Debug)
        BUILD_TYPE="Debug"
        ;;
    release|Release)
        BUILD_TYPE="Release"
        ;;
    rebuild)
        ACTION="rebuild"
        ;;
    clean)
        ACTION="clean"
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 1
        ;;
esac

build_project() {
    if [[ "${ACTION}" == "clean" ]]; then
        rm -rf "${BUILD_DIR}"
        echo "Removed ${BUILD_DIR}"
        return
    fi

    if [[ "${ACTION}" == "rebuild" ]]; then
        rm -rf "${BUILD_DIR}"
    fi

    mkdir -p "${ROOT_DIR}/assets/shaders"

    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

    cmake --build "${BUILD_DIR}" --parallel

    echo
    echo "Build complete: ${BUILD_DIR}/CH_MapViewer"
}

if [[ -n "${IN_NIX_SHELL:-}" ]]; then
    build_project
else
    command -v nix >/dev/null 2>&1 || {
        echo "Error: nix was not found in PATH."
        exit 1
    }

    export ROOT_DIR BUILD_DIR BUILD_TYPE ACTION
    export -f build_project
    nix develop "${ROOT_DIR}" --command bash -c 'set -euo pipefail; build_project'
fi
