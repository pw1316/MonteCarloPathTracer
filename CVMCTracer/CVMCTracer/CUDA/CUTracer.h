#pragma once
#include <cuda_runtime.h>
#include "Framework/ObjReader.hpp"

namespace PW
{
    namespace Tracer
    {
        cudaError_t RenderScene1(const FileReader::ObjModel *model, PWVector4f *hostcolor);
    }
}
