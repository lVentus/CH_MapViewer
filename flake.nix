{
  description = "CH_MapViewer development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            gcc
            git
            python3
            wayland-scanner
          ];

          buildInputs = with pkgs; [
            mesa
            libGL

            wayland
            wayland-protocols
            libxkbcommon

            libx11
            libxrandr
            libxinerama
            libxcursor
            libxi
            xorgproto
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.libGL
            pkgs.mesa
            pkgs.wayland
            pkgs.libxkbcommon
            pkgs.libx11
            pkgs.libxrandr
            pkgs.libxinerama
            pkgs.libxcursor
            pkgs.libxi
          ];

          CMAKE_GENERATOR = "Ninja";
        };
      });
}
