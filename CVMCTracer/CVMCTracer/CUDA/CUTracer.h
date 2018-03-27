#pragma once
#include <stdafx.h>
#include "Framework/ObjReader.hpp"

namespace PW
{
    namespace Tracer
    {
        cudaError_t Initialize();
        cudaError_t CreateGeometry(const PW::FileReader::ObjModel *model);
        cudaError_t DestroyGeometry();
        cudaError_t RenderScene(const PWint sceneID, PWVector3f *hostcolor);
    }
}
