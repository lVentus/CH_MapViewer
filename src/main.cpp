#include "core/Application.h"

#include <exception>
#include <filesystem>
#include <iostream>

namespace {

std::filesystem::path FindAssetDirectory(const char* executablePath) {
    const auto workingAssets = std::filesystem::current_path() / "assets";
    if (std::filesystem::is_directory(workingAssets)) {
        return workingAssets;
    }

    return std::filesystem::absolute(executablePath).parent_path() / "assets";
}

} // namespace

int main(int argc, char** argv) {
    try {
        chmv::core::Application app(FindAssetDirectory(argv[0]));

        if (argc == 3) {
            app.LoadDataset(argv[1], argv[2]);
        } else if (argc != 1) {
            std::cerr << "Usage: CH_MapViewer [graph.sch graph.sch.ranges]\n";
            return 1;
        }

        return app.Run();
    } catch (const std::exception& error) {
        std::cerr << "CH_MapViewer: " << error.what() << '\n';
        return 1;
    }
}
