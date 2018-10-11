#pragma once
#include <stdafx.h>

namespace Quin::RTX
{
    class Shader
    {
    public:
        Shader(ID3D11Device* device)
        {
            HRESULT hr = S_OK;
            ID3D10Blob* blob = nullptr;
            ID3D10Blob* errblob = nullptr;
            hr = D3DX11CompileFromFile("Shader/rtx.hlsl", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_SKIP_OPTIMIZATION, 0, nullptr, &blob, &errblob, nullptr);
            if (errblob)
            {
                OutputDebugString((char*)errblob->GetBufferPointer());
            }
            FAILTHROW;
            hr = device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs_rtx);
            FAILTHROW;
            SafeRelease(&blob);
        }
        ~Shader()
        {
            SafeRelease(&cs_rtx);
        }
        ID3D11ComputeShader* cs_rtx = nullptr;
    };
}
