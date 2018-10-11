#pragma once
#include <stdafx.h>

#include <vector>

#include <Utils/Structure.hpp>

namespace Quin::RTX
{
    class ShaderResource
    {
    public:
        ShaderResource(ID3D11Device* device, const Utils::Model& model, const UINT width, const UINT height)
        {
            HRESULT hr = S_OK;

            const UINT VN = static_cast<UINT>(model.attr.vertices.size()) / 3U;
            const UINT NN = static_cast<UINT>(model.attr.normals.size()) / 3U;
            const UINT GN = static_cast<UINT>(model.shapes.size());
            const UINT MN = static_cast<UINT>(model.materials.size());

            ID3D11Buffer* buffer = nullptr;
            ID3D11Texture2D* texture2D = nullptr;
            D3D11_BUFFER_DESC bufferDesc;
            D3D11_TEXTURE2D_DESC texture2DDesc;
            D3D11_SUBRESOURCE_DATA subData;
            D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
            D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;

            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(Utils::CB0);
            bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
            bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            bufferDesc.MiscFlags = 0;
            bufferDesc.StructureByteStride = 0;
            device->CreateBuffer(&bufferDesc, nullptr, &cb0);
            FAILTHROW;

            /* vertex */
            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(D3DXVECTOR3) * VN;
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bufferDesc.StructureByteStride = sizeof(D3DXVECTOR3);
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &model.attr.vertices[0];
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &buffer);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.FirstElement = 0U;
            srvDesc.Buffer.NumElements = VN;
            device->CreateShaderResourceView(buffer, &srvDesc, &vertex);
            FAILTHROW;
            SafeRelease(&buffer);

            /* normal */
            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(D3DXVECTOR3) * NN;
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bufferDesc.StructureByteStride = sizeof(D3DXVECTOR3);
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &model.attr.normals[0];
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &buffer);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.FirstElement = 0U;
            srvDesc.Buffer.NumElements = NN;
            device->CreateShaderResourceView(buffer, &srvDesc, &normal);
            FAILTHROW;
            SafeRelease(&buffer);

            /* Geometry & Triangle */
            std::vector<Utils::CSTriangle> tries;
            for (auto& shape : model.shapes)
            {
                auto numTries = static_cast<UINT>(shape.mesh.material_ids.size());
                for (UINT i = 0; i < numTries; ++i)
                {
                    Utils::CSTriangle tri;
                    tri.v0 = shape.mesh.indices[3 * i + 0].vertex_index;
                    tri.v1 = shape.mesh.indices[3 * i + 1].vertex_index;
                    tri.v2 = shape.mesh.indices[3 * i + 2].vertex_index;
                    tri.n0 = shape.mesh.indices[3 * i + 0].normal_index;
                    tri.n1 = shape.mesh.indices[3 * i + 1].normal_index;
                    tri.n2 = shape.mesh.indices[3 * i + 2].normal_index;
                    tri.matId = shape.mesh.material_ids[i];
                    tries.push_back(std::move(tri));
                }
            }
            const UINT TN = static_cast<UINT>(tries.size());
            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(Utils::CSTriangle) * TN;
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bufferDesc.StructureByteStride = sizeof(Utils::CSTriangle);
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &tries[0];
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &buffer);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.FirstElement = 0U;
            srvDesc.Buffer.NumElements = TN;
            device->CreateShaderResourceView(buffer, &srvDesc, &triangle);
            FAILTHROW;
            SafeRelease(&buffer);

            /* Material */
            std::vector<Utils::CSMaterial> mats;
            for (auto& mat : model.materials)
            {
                Utils::CSMaterial csmat;
                csmat.Ka = mat.ambient;
                csmat.Kd = mat.diffuse;
                csmat.Ks = mat.specular;
                csmat.Ns = mat.shininess;
                csmat.Tr = 1.0f - mat.dissolve;
                csmat.Ni = mat.ior;
                mats.push_back(std::move(csmat));
            }
            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(Utils::CSMaterial) * MN;
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bufferDesc.StructureByteStride = sizeof(Utils::CSMaterial);
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &mats[0];
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &buffer);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.FirstElement = 0U;
            srvDesc.Buffer.NumElements = MN;
            device->CreateShaderResourceView(buffer, &srvDesc, &material);
            FAILTHROW;
            SafeRelease(&buffer);

            /* RTV */
            ZeroMemory(&texture2DDesc, sizeof(texture2DDesc));
            texture2DDesc.Width = width;
            texture2DDesc.Height = height;
            texture2DDesc.MipLevels = 1;
            texture2DDesc.ArraySize = 1;
            texture2DDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            texture2DDesc.SampleDesc.Count = 1;
            texture2DDesc.SampleDesc.Quality = 0;
            texture2DDesc.Usage = D3D11_USAGE_DEFAULT;
            texture2DDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            texture2DDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
            texture2DDesc.MiscFlags = 0;
            device->CreateTexture2D(&texture2DDesc, nullptr, &texture2D);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MostDetailedMip = 0;
            srvDesc.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(texture2D, &srvDesc, &screen_r);
            FAILTHROW;
            SafeRelease(&texture2D);

            ZeroMemory(&texture2DDesc, sizeof(texture2DDesc));
            texture2DDesc.Width = width;
            texture2DDesc.Height = height;
            texture2DDesc.MipLevels = 1;
            texture2DDesc.ArraySize = 1;
            texture2DDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            texture2DDesc.SampleDesc.Count = 1;
            texture2DDesc.SampleDesc.Quality = 0;
            texture2DDesc.Usage = D3D11_USAGE_DEFAULT;
            texture2DDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
            texture2DDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
            texture2DDesc.MiscFlags = 0;
            device->CreateTexture2D(&texture2DDesc, nullptr, &texture2D);
            FAILTHROW;
            ZeroMemory(&uavDesc, sizeof(uavDesc));
            uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            uavDesc.Texture2D.MipSlice = 0;
            device->CreateUnorderedAccessView(texture2D, &uavDesc, &screen_w);
            FAILTHROW;
            SafeRelease(&texture2D);

            ZeroMemory(&texture2DDesc, sizeof(texture2DDesc));
            texture2DDesc.Width = width;
            texture2DDesc.Height = height;
            texture2DDesc.MipLevels = 1;
            texture2DDesc.ArraySize = 1;
            texture2DDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            texture2DDesc.SampleDesc.Count = 1;
            texture2DDesc.SampleDesc.Quality = 0;
            texture2DDesc.Usage = D3D11_USAGE_DEFAULT;
            texture2DDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
            texture2DDesc.CPUAccessFlags = 0;
            texture2DDesc.MiscFlags = 0;
            device->CreateTexture2D(&texture2DDesc, nullptr, &texture2D);
            FAILTHROW;
            ZeroMemory(&uavDesc, sizeof(uavDesc));
            uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            uavDesc.Texture2D.MipSlice = 0;
            device->CreateUnorderedAccessView(texture2D, &uavDesc, &rtv);
            FAILTHROW;
            SafeRelease(&texture2D);
        }
        ~ShaderResource()
        {
            SafeRelease(&screen_w);
            SafeRelease(&screen_r);
            SafeRelease(&material);
            SafeRelease(&triangle);
            SafeRelease(&normal);
            SafeRelease(&vertex);
            SafeRelease(&cb0);
        }
        ID3D11Buffer* cb0 = nullptr;

        ID3D11ShaderResourceView* vertex = nullptr;
        ID3D11ShaderResourceView* normal = nullptr;
        ID3D11ShaderResourceView* triangle = nullptr;
        ID3D11ShaderResourceView* material = nullptr;

        ID3D11ShaderResourceView* screen_r = nullptr;
        ID3D11UnorderedAccessView* screen_w = nullptr;
        ID3D11UnorderedAccessView* rtv = nullptr;
    private:
    };
}
