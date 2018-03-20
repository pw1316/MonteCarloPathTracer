#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Framework/ObjReader.hpp"

namespace PW
{
    namespace Tracer
    {
        cudaError_t RenderScene1(FileReader::ObjModel *model);
    }
}
