#pragma once
#include <stdafx.h>

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
            //const UINT TN = static_cast<UINT>(model.attr.vertices.size()) / 3U;
            //const UINT GN = static_cast<UINT>(model.attr.vertices.size()) / 3U;

            ID3D11Buffer* buffer = nullptr;
            D3D11_BUFFER_DESC bufferDesc;
            D3D11_SUBRESOURCE_DATA subData;
            D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

            Utils::CBGeometry cbGeo;
            cbGeo.nVertex = VN;
            cbGeo.nNormal = NN;
            //cbGeo.nTriangle = model.attr.vertices.size() / 3U;
            //cbGeo.nNormal = model.attr.normals.size() / 3U;
            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(Utils::CBGeometry);
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = 0;
            bufferDesc.StructureByteStride = 0;
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &cbGeo;
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &cbGeometry);
            FAILTHROW;

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
        }
        ~ShaderResource()
        {
            SafeRelease(&geometry);
            SafeRelease(&triangle);
            SafeRelease(&normal);
            SafeRelease(&vertex);
            SafeRelease(&cbGeometry);
        }
        ID3D11Buffer* cbGeometry = nullptr;
        ID3D11ShaderResourceView* vertex = nullptr;
        ID3D11ShaderResourceView* normal = nullptr;
        ID3D11ShaderResourceView* triangle = nullptr;
        ID3D11ShaderResourceView* geometry = nullptr;
    private:
    };
}
