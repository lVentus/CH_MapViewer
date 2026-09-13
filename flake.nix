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
        pythonEnv = pkgs.python3.withPackages (pythonPackages: with pythonPackages; [
          jinja2
        ]);
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            gcc
            git
            pythonEnv
            wayland-scanner
          ];

          buildInputs = with pkgs; [
            mesa
            libGL
            libffi

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

          # CMake's FindPython resolves the pythonEnv interpreter symlink to the
          # base Python store path. Keep the environment's site-packages visible
          # so GLAD's generator can still import Jinja2.
          PYTHONPATH = "${pythonEnv}/${pkgs.python3.sitePackages}";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.libGL
            pkgs.mesa
            pkgs.libffi
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
