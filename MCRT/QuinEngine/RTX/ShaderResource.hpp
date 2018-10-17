#pragma once
#include <stdafx.h>

#include <vector>

#include <Utils/Structure.hpp>
#include <Utils/KDTree.hpp>

namespace Quin::RTX
{
    class ShaderResourceManager
    {
    public:
        ShaderResourceManager(ID3D11Device* device, const Utils::QuinModel& model, const Utils::KDTree& kdtree, const UINT width, const UINT height)
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
            device->CreateBuffer(&bufferDesc, nullptr, &cb_0);
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
            device->CreateShaderResourceView(buffer, &srvDesc, &srvVertex);
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
            device->CreateShaderResourceView(buffer, &srvDesc, &srvNormal);
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
            device->CreateShaderResourceView(buffer, &srvDesc, &srvTriangle);
            FAILTHROW;
            SafeRelease(&buffer);

            std::vector<Utils::CSKDTree> tree;
            {
                std::list<Utils::KDNode*> bfs;
                bfs.push_back(kdtree.m_root);
                while (!bfs.empty())
                {
                    auto node = bfs.front();
                    bfs.pop_front();
                    Utils::CSKDTree csnode;
                    if (node->split.axis != Utils::KDAxis::AXIS_NONE)
                    {
                        assert(node->left && node->right);
                        csnode.left = static_cast<UINT>(bfs.size() + tree.size() + 1);
                        csnode.right = static_cast<UINT>(bfs.size() + tree.size() + 2);
                        csnode.aabbMin.x = node->aabb.min()[0];
                        csnode.aabbMin.y = node->aabb.min()[1];
                        csnode.aabbMin.z = node->aabb.min()[2];
                        csnode.aabbMax.x = node->aabb.max()[0];
                        csnode.aabbMax.y = node->aabb.max()[1];
                        csnode.aabbMax.z = node->aabb.max()[2];
                        csnode.splitAxis = static_cast<UINT>(node->split.axis);
                        csnode.splitValue = node->split.value;
                        csnode.numTri = 0U;
                        bfs.push_back(node->left);
                        bfs.push_back(node->right);
                    }
                    else
                    {
                        assert(!node->left && !node->right);
                        csnode.left = -1;
                        csnode.right = -1;
                        csnode.aabbMin.x = node->aabb.min()[0];
                        csnode.aabbMin.y = node->aabb.min()[1];
                        csnode.aabbMin.z = node->aabb.min()[2];
                        csnode.aabbMax.x = node->aabb.max()[0];
                        csnode.aabbMax.y = node->aabb.max()[1];
                        csnode.aabbMax.z = node->aabb.max()[2];
                        csnode.splitAxis = 0U;
                        csnode.splitValue = 0.0f;
                        csnode.numTri = 0U;
                        for (auto triId : node->triangleIds)
                        {
                            csnode.triIds[csnode.numTri++] = triId;
                        }
                    }
                    for (UINT i = csnode.numTri; i < 64; ++i)
                    {
                        csnode.triIds[i] = -1;
                    }
                    tree.push_back(std::move(csnode));
                }
            }
            /* KDTree */
            ZeroMemory(&bufferDesc, sizeof(bufferDesc));
            bufferDesc.ByteWidth = sizeof(Utils::CSKDTree) * static_cast<UINT>(tree.size());
            bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
            bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            bufferDesc.CPUAccessFlags = 0;
            bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            bufferDesc.StructureByteStride = sizeof(Utils::CSKDTree);
            ZeroMemory(&subData, sizeof(subData));
            subData.pSysMem = &tree[0];
            subData.SysMemPitch = 0;
            subData.SysMemSlicePitch = 0;
            device->CreateBuffer(&bufferDesc, &subData, &buffer);
            FAILTHROW;
            ZeroMemory(&srvDesc, sizeof(srvDesc));
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.FirstElement = 0U;
            srvDesc.Buffer.NumElements = static_cast<UINT>(tree.size());
            device->CreateShaderResourceView(buffer, &srvDesc, &srvKDTree);
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
            device->CreateShaderResourceView(buffer, &srvDesc, &srvMaterial);
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
            device->CreateShaderResourceView(texture2D, &srvDesc, &srvScreen);
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
            device->CreateUnorderedAccessView(texture2D, &uavDesc, &uavScreen);
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
            device->CreateUnorderedAccessView(texture2D, &uavDesc, &uavRenderTarget);
            FAILTHROW;
            SafeRelease(&texture2D);
        }
        ~ShaderResourceManager()
        {
            SafeRelease(&uavRenderTarget);
            SafeRelease(&uavScreen);
            SafeRelease(&srvScreen);
            SafeRelease(&srvMaterial);
            SafeRelease(&srvKDTree);
            SafeRelease(&srvTriangle);
            SafeRelease(&srvNormal);
            SafeRelease(&srvVertex);
            SafeRelease(&cb_0);
        }
        ID3D11Buffer* cb_0 = nullptr;

        ID3D11ShaderResourceView* srvVertex = nullptr;
        ID3D11ShaderResourceView* srvNormal = nullptr;
        ID3D11ShaderResourceView* srvTriangle = nullptr;
        ID3D11ShaderResourceView* srvKDTree = nullptr;
        ID3D11ShaderResourceView* srvMaterial = nullptr;

        ID3D11ShaderResourceView* srvScreen = nullptr;
        ID3D11UnorderedAccessView* uavScreen = nullptr;
        ID3D11UnorderedAccessView* uavRenderTarget = nullptr;
    private:
    };
}
