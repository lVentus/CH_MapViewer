#include "core/Application.h"

#include <exception>
#include <filesystem>
#include <iostream>

namespace {

std::filesystem::path FindAssetDirectory(const char* executablePath) {
    const auto workingAssets = std::filesystem::current_path() / "assets";
    if (std::filesystem::is_directory(workingAssets / "shaders") &&
        std::filesystem::is_directory(workingAssets / "data")) {
        return workingAssets;
    }

    auto directory = std::filesystem::absolute(executablePath).parent_path();
    std::filesystem::path shaderFallback;
    while (!directory.empty()) {
        const auto assets = directory / "assets";
        if (std::filesystem::is_directory(assets / "shaders")) {
            if (std::filesystem::is_directory(assets / "data")) {
                return assets;
            }
            if (shaderFallback.empty()) {
                shaderFallback = assets;
            }
        }

        const auto parent = directory.parent_path();
        if (parent == directory) {
            break;
        }
        directory = parent;
    }

    if (!shaderFallback.empty()) {
        return shaderFallback;
    }

    return std::filesystem::absolute(executablePath).parent_path() / "assets";
}

} // namespace

int main(int argc, char** argv) {
    try {
        chmv::core::Application app(FindAssetDirectory(argv[0]));

        if (argc == 3) {
            chmv::data::DatasetEntry dataset;
            dataset.graphPath = argv[1];
            dataset.rangesPath = argv[2];
            app.LoadDataset(dataset);
        } else if (argc == 2 && std::filesystem::path(argv[1]).extension() == ".chlod") {
            chmv::data::DatasetEntry dataset;
            dataset.pageFilePath = argv[1];
            app.LoadDataset(dataset);
        } else if (argc != 1) {
            std::cerr << "Usage: CH_MapViewer [graph.sch graph.sch.ranges | graph.chlod]\n";
            return 1;
        }

        return app.Run();
    } catch (const std::exception& error) {
        std::cerr << "CH_MapViewer: " << error.what() << '\n';
        return 1;
    }
}
