#pragma once
#include <stdafx.h>
#include "Framework/ObjReader.hpp"

namespace PW
{
    namespace Tracer
    {
        cudaError_t RenderScene1(const FileReader::ObjModel *model, PWVector4f *hostcolor);
    }
}
