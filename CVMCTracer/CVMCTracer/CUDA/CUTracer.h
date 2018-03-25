#pragma once
#include <stdafx.h>
#include "Framework/ObjReader.hpp"

namespace PW
{
    namespace Tracer
    {
        cudaError_t RenderScene(const PWint sceneID, const FileReader::ObjModel *model, PWVector4f *hostcolor);
    }
}
