#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-linux"
BUILD_TYPE="Debug"
ACTION="build"
ORIGINAL_ARG="${1:-}"
BUILD_APP="${CHMV_BUILD_APP:-ON}"
BUILD_TESTS="${CHMV_BUILD_TESTS:-ON}"

usage() {
    cat <<'USAGE'
Usage:
  ./build.sh
  ./build.sh debug
  ./build.sh release
  ./build.sh rebuild
  ./build.sh clean

Linux behavior:
  - NixOS / nix shell: uses flake.nix via `nix develop`.
  - Ubuntu / Debian: installs missing build, OpenGL, X11 and Wayland packages with apt.
  - Fedora/RHEL family: installs missing packages with dnf.
  - Arch family: installs missing packages with pacman.
  - openSUSE/SLES: installs missing packages with zypper.

Environment overrides:
  CHMV_BUILD_APP=OFF       Build only the non-graphics/core targets.
  CHMV_BUILD_TESTS=OFF     Do not build the test executable.
  CHMV_USE_NIX=1           Force `nix develop` on a non-NixOS distribution.

Build output:
  build-linux/CH_MapViewer
USAGE
}

case "${ORIGINAL_ARG}" in
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

run_as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
        return
    fi

    echo "Error: installing system dependencies requires root privileges or sudo." >&2
    exit 1
}

append_graphics_deps_if_needed() {
    local base_name="$1"
    local graphics_name="$2"
    local -n base_ref="${base_name}"
    local -n graphics_ref="${graphics_name}"

    if [[ "${BUILD_APP^^}" != "OFF" ]]; then
        base_ref+=("${graphics_ref[@]}")
    fi
}

install_debian_dependencies() {
    local packages=(
        build-essential
        cmake
        ninja-build
        git
    )
    local graphics_packages=(
        pkg-config
        python3
        python3-jinja2
        libgl1-mesa-dev
        libx11-dev
        libxrandr-dev
        libxinerama-dev
        libxcursor-dev
        libxi-dev
        libwayland-dev
        wayland-protocols
        libxkbcommon-dev
    )
    append_graphics_deps_if_needed packages graphics_packages

    local missing=()
    local package
    for package in "${packages[@]}"; do
        if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q '^install ok installed$'; then
            missing+=("${package}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        echo "System dependencies: already installed (Debian/Ubuntu)."
        return
    fi

    echo "Installing missing Debian/Ubuntu dependencies:"
    printf '  %s\n' "${missing[@]}"
    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
}

install_fedora_dependencies() {
    local packages=(
        gcc
        gcc-c++
        cmake
        ninja-build
        git
    )
    local graphics_packages=(
        pkgconf-pkg-config
        python3
        python3-jinja2
        mesa-libGL-devel
        libX11-devel
        libXrandr-devel
        libXinerama-devel
        libXcursor-devel
        libXi-devel
        wayland-devel
        wayland-protocols-devel
        libxkbcommon-devel
    )
    append_graphics_deps_if_needed packages graphics_packages

    local missing=()
    local package
    for package in "${packages[@]}"; do
        if ! rpm -q "${package}" >/dev/null 2>&1; then
            missing+=("${package}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        echo "System dependencies: already installed (Fedora/RHEL family)."
        return
    fi

    local dnf_cmd="dnf"
    if command -v dnf5 >/dev/null 2>&1; then
        dnf_cmd="dnf5"
    fi

    command -v "${dnf_cmd}" >/dev/null 2>&1 || {
        echo "Error: ${dnf_cmd} was not found." >&2
        exit 1
    }

    echo "Installing missing Fedora/RHEL dependencies:"
    printf '  %s\n' "${missing[@]}"
    run_as_root "${dnf_cmd}" install -y "${missing[@]}"
}

install_arch_dependencies() {
    local packages=(
        base-devel
        cmake
        ninja
        git
    )
    local graphics_packages=(
        pkgconf
        python
        python-jinja
        mesa
        libx11
        libxrandr
        libxinerama
        libxcursor
        libxi
        wayland
        wayland-protocols
        libxkbcommon
    )
    append_graphics_deps_if_needed packages graphics_packages

    local missing=()
    local package
    for package in "${packages[@]}"; do
        if ! pacman -Q "${package}" >/dev/null 2>&1; then
            missing+=("${package}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        echo "System dependencies: already installed (Arch family)."
        return
    fi

    echo "Installing missing Arch dependencies:"
    printf '  %s\n' "${missing[@]}"
    run_as_root pacman -S --needed --noconfirm "${missing[@]}"
}

install_opensuse_dependencies() {
    local packages=(
        gcc
        gcc-c++
        cmake
        ninja
        git
    )
    local graphics_packages=(
        pkg-config
        python3
        python3-Jinja2
        Mesa-libGL-devel
        libX11-devel
        libXrandr-devel
        libXinerama-devel
        libXcursor-devel
        libXi-devel
        wayland-devel
        wayland-protocols-devel
        libxkbcommon-devel
    )
    append_graphics_deps_if_needed packages graphics_packages

    local missing=()
    local package
    for package in "${packages[@]}"; do
        if ! rpm -q "${package}" >/dev/null 2>&1; then
            missing+=("${package}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        echo "System dependencies: already installed (openSUSE/SLES)."
        return
    fi

    echo "Installing missing openSUSE/SLES dependencies:"
    printf '  %s\n' "${missing[@]}"
    run_as_root zypper --non-interactive install "${missing[@]}"
}

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
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCHMV_BUILD_APP="${BUILD_APP}" \
        -DCHMV_BUILD_TESTS="${BUILD_TESTS}"

    cmake --build "${BUILD_DIR}" --parallel

    echo
    if [[ "${BUILD_APP^^}" == "OFF" ]]; then
        echo "Build complete: graphics application disabled (CHMV_BUILD_APP=OFF)."
    else
        echo "Build complete: ${BUILD_DIR}/CH_MapViewer"
    fi
}

enter_nix_and_build() {
    command -v nix >/dev/null 2>&1 || {
        echo "Error: this build path requires Nix, but 'nix' was not found in PATH." >&2
        exit 1
    }

    echo "Using flake.nix development environment."
    if [[ -n "${ORIGINAL_ARG}" ]]; then
        nix develop "${ROOT_DIR}" --command bash "${ROOT_DIR}/build.sh" "${ORIGINAL_ARG}"
    else
        nix develop "${ROOT_DIR}" --command bash "${ROOT_DIR}/build.sh"
    fi
}

# Cleaning does not need a compiler or system packages.
if [[ "${ACTION}" == "clean" ]]; then
    build_project
    exit 0
fi

# Once inside the flake shell, all dependencies are already supplied by Nix.
if [[ -n "${IN_NIX_SHELL:-}" ]]; then
    build_project
    exit 0
fi

if [[ "${CHMV_USE_NIX:-0}" == "1" ]]; then
    enter_nix_and_build
    exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: build.sh is intended for Linux. Use the Windows CMake instructions from README.md on Windows." >&2
    exit 1
fi

if [[ ! -r /etc/os-release ]]; then
    echo "Error: /etc/os-release was not found, so the Linux distribution could not be detected." >&2
    echo "Install the dependencies listed in README.md and run CMake manually." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
DISTRO_ID="${ID:-unknown}"
DISTRO_LIKE="${ID_LIKE:-}"
DISTRO_WORDS=" ${DISTRO_ID} ${DISTRO_LIKE} "

echo "Detected Linux distribution: ${PRETTY_NAME:-${DISTRO_ID}}"

if [[ "${DISTRO_ID}" == "nixos" ]]; then
    enter_nix_and_build
elif [[ "${DISTRO_WORDS}" == *" debian "* || "${DISTRO_WORDS}" == *" ubuntu "* ]]; then
    install_debian_dependencies
    build_project
elif [[ "${DISTRO_WORDS}" == *" fedora "* || "${DISTRO_WORDS}" == *" rhel "* || \
        "${DISTRO_WORDS}" == *" centos "* || "${DISTRO_WORDS}" == *" rocky "* || \
        "${DISTRO_WORDS}" == *" almalinux "* ]]; then
    install_fedora_dependencies
    build_project
elif [[ "${DISTRO_WORDS}" == *" arch "* || "${DISTRO_WORDS}" == *" manjaro "* ]]; then
    install_arch_dependencies
    build_project
elif [[ "${DISTRO_WORDS}" == *" opensuse "* || "${DISTRO_WORDS}" == *" suse "* || \
        "${DISTRO_ID}" == opensuse* || "${DISTRO_ID}" == "sles" ]]; then
    install_opensuse_dependencies
    build_project
elif command -v nix >/dev/null 2>&1; then
    echo "Distribution is not explicitly supported by build.sh; Nix is available, so using flake.nix."
    enter_nix_and_build
else
    echo "Error: unsupported Linux distribution '${DISTRO_ID}'." >&2
    echo "Install the dependencies listed in README.md and build with CMake manually," >&2
    echo "or install Nix and run CHMV_USE_NIX=1 ./build.sh." >&2
    exit 1
fi
