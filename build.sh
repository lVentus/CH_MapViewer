#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-linux"
BUILD_TYPE="Debug"
ACTION="build"

usage() {
    cat <<'EOF'
Usage:
  ./build_linux.sh
  ./build_linux.sh debug
  ./build_linux.sh release
  ./build_linux.sh rebuild
  ./build_linux.sh clean

Behavior:
  NixOS  : uses the project's flake via `nix develop`
  Ubuntu : installs required build dependencies with apt, then builds

Build output:
  build-linux/CH_MapViewer
EOF
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

ensure_runtime_dirs() {
    # The current CMake setup copies this directory after linking.
    # Keep it present even if shaders are supplied separately.
    mkdir -p "${ROOT_DIR}/assets/shaders"
}

configure_and_build() {
    if [[ "${ACTION}" == "clean" ]]; then
        rm -rf "${BUILD_DIR}"
        echo "Removed ${BUILD_DIR}"
        return
    fi

    if [[ "${ACTION}" == "rebuild" ]]; then
        rm -rf "${BUILD_DIR}"
    fi

    ensure_runtime_dirs

    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

    cmake --build "${BUILD_DIR}" --parallel

    echo
    echo "Build complete: ${BUILD_DIR}/CH_MapViewer"
}

run_nixos() {
    command -v nix >/dev/null 2>&1 || {
        echo "Error: NixOS detected but 'nix' is not available in PATH."
        exit 1
    }

    if [[ ! -f "${ROOT_DIR}/flake.nix" ]]; then
        echo "Error: ${ROOT_DIR}/flake.nix was not found."
        exit 1
    fi

    if [[ -n "${IN_NIX_SHELL:-}" ]]; then
        configure_and_build
        return
    fi

    export ROOT_DIR BUILD_DIR BUILD_TYPE ACTION
    export -f ensure_runtime_dirs configure_and_build

    nix develop "${ROOT_DIR}" --command bash -c 'configure_and_build'
}

install_ubuntu_dependencies() {
    local sudo_cmd=()

    if [[ "${EUID}" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || {
            echo "Error: Ubuntu dependency installation requires sudo."
            exit 1
        }
        sudo_cmd=(sudo)
    fi

    echo "Installing Ubuntu build dependencies..."
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        python3 \
        libgl1-mesa-dev \
        libwayland-dev \
        libwayland-bin \
        wayland-protocols \
        libxkbcommon-dev \
        xorg-dev
}

run_ubuntu() {
    if [[ "${ACTION}" != "clean" ]]; then
        install_ubuntu_dependencies
    fi
    configure_and_build
}

if [[ ! -r /etc/os-release ]]; then
    echo "Error: could not detect Linux distribution (/etc/os-release missing)."
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

case "${ID:-}" in
    nixos)
        run_nixos
        ;;
    ubuntu)
        run_ubuntu
        ;;
    *)
        echo "Unsupported distribution: ${PRETTY_NAME:-${ID:-unknown}}"
        echo "This script currently supports NixOS and Ubuntu."
        exit 1
        ;;
esac
