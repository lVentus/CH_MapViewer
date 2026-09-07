#pragma once

namespace chmv::pipeline {

enum class ProcessingMode {
    GPUDriven,
    CPUReference,
    Validation,
};

} // namespace chmv::pipeline
