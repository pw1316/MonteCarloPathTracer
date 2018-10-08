#pragma once
#include <stdafx.h>

#include <vector>

#include <Utils/Structure.hpp>

namespace Quin::RTX
{
    class ShaderResource
    {
    public:
        ShaderResource(ID3D11Device* device, const Utils::Model& model)
        {
            HRESULT hr = S_OK;

            const UINT VN = static_cast<UINT>(model.attr.vertices.size()) / 3U;
            const UINT NN = static_cast<UINT>(model.attr.normals.size()) / 3U;
            const UINT GN = static_cast<UINT>(model.shapes.size());

            ID3D11Buffer* buffer = nullptr;
            D3D11_BUFFER_DESC bufferDesc;
            D3D11_SUBRESOURCE_DATA subData;
            D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

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
            std::vector<Utils::CSGeometry> geoes;
            for (auto& shape : model.shapes)
            {
                Utils::CSGeometry geo;
                geo.startTri = tries.size();
                geo.numTries = shape.mesh.material_ids.size();
                for (UINT i = 0; i < geo.numTries; ++i)
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
                geoes.push_back(std::move(geo));
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

            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(Utils::CSGeometry) * GN;
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bufferDesc.StructureByteStride = sizeof(Utils::CSGeometry);
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &geoes[0];
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &buffer);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.FirstElement = 0U;
            srvDesc.Buffer.NumElements = GN;
            device->CreateShaderResourceView(buffer, &srvDesc, &geometry);
            FAILTHROW;
            SafeRelease(&buffer);
        }
        ~ShaderResource()
        {
            SafeRelease(&geometry);
            SafeRelease(&triangle);
            SafeRelease(&normal);
            SafeRelease(&vertex);
        }
        ID3D11ShaderResourceView* vertex = nullptr;
        ID3D11ShaderResourceView* normal = nullptr;
        ID3D11ShaderResourceView* triangle = nullptr;
        ID3D11ShaderResourceView* geometry = nullptr;
    private:
    };
}
