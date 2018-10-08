#include <stdafx.h>
#include "Font.hpp"

#include <fstream>

void Quin::System::DX11::Font::Initialize(ID3D11Device * device)
{
    HRESULT hr = S_OK;
    std::ifstream fontMeta("Res/font_meta.txt");
    hr = fontMeta ? S_OK : E_FAIL;
    FAILTHROW;

    for (int i = 0; i < 95; ++i)
    {
        int dummy;
        fontMeta >> dummy;
        fontMeta >> m_Font[i].left;
        fontMeta >> m_Font[i].right;
        fontMeta >> m_Font[i].size;
    }
    fontMeta.close();

    InitializeBuffer(device);
    InitializeShader(device);
}

void Quin::System::DX11::Font::Shutdown()
{
    ShutdownShader();
    ShutdownBuffer();
}

void Quin::System::DX11::Font::Render(ID3D11Device* device, ID3D11DeviceContext* context, const std::string& text, const D3DXVECTOR2& pos, const UINT w, const UINT h, const D3DXMATRIX& proj)
{
    HRESULT hr = S_OK;

    D3D11_BUFFER_DESC bufferDesc;
    D3D11_SUBRESOURCE_DATA subData;
    D3D11_MAPPED_SUBRESOURCE mapped{};

    UINT VN = static_cast<UINT>(text.size() * 6);
    VBType *vertices = new VBType[VN];
    ULONG *indices = new ULONG[VN];
    float x = -(w * 0.5f) + pos.x;
    float y = h * 0.5f - pos.y;
    for (UINT vId = 0; vId < VN / 6; ++vId)
    {
        auto letter = text[vId] - 32;
        if (letter == 0)
        {
            x += 3.0f;
        }
        else
        {
            vertices[vId * 6 + 0].pos = D3DXVECTOR3(x, y, 1.0f);
            vertices[vId * 6 + 0].uv = D3DXVECTOR2(m_Font[letter].left, 0.0f);
            vertices[vId * 6 + 1].pos = D3DXVECTOR3((x + m_Font[letter].size), (y - 16), 1.0f);
            vertices[vId * 6 + 1].uv = D3DXVECTOR2(m_Font[letter].right, 1.0f);
            vertices[vId * 6 + 2].pos = D3DXVECTOR3(x, (y - 16), 1.0f);
            vertices[vId * 6 + 2].uv = D3DXVECTOR2(m_Font[letter].left, 1.0f);

            vertices[vId * 6 + 3].pos = D3DXVECTOR3(x, y, 1.0f);
            vertices[vId * 6 + 3].uv = D3DXVECTOR2(m_Font[letter].left, 0.0f);
            vertices[vId * 6 + 4].pos = D3DXVECTOR3(x + m_Font[letter].size, y, 1.0f);
            vertices[vId * 6 + 4].uv = D3DXVECTOR2(m_Font[letter].right, 0.0f);
            vertices[vId * 6 + 5].pos = D3DXVECTOR3((x + m_Font[letter].size), (y - 16), 1.0f);
            vertices[vId * 6 + 5].uv = D3DXVECTOR2(m_Font[letter].right, 1.0f);
            x += m_Font[letter].size + 1.0f;
        }
    }
    for (UINT vId = 0; vId < VN; ++vId)
    {
        indices[vId] = vId;
    }

    SafeRelease(&m_VB);
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
    bufferDesc.ByteWidth = sizeof(VBType) * VN;
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 0;
    ZeroMemory(&subData, sizeof(subData));
    subData.pSysMem = vertices;
    subData.SysMemPitch = 0;
    subData.SysMemSlicePitch = 0;
    hr = device->CreateBuffer(&bufferDesc, &subData, &m_VB);
    FAILTHROW;

    SafeRelease(&m_IB);
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
    bufferDesc.ByteWidth = sizeof(ULONG) * VN;
    bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 0;
    ZeroMemory(&subData, sizeof(subData));
    subData.pSysMem = indices;
    subData.SysMemPitch = 0;
    subData.SysMemSlicePitch = 0;
    hr = device->CreateBuffer(&bufferDesc, &subData, &m_IB);
    FAILTHROW;
    delete[] vertices; vertices = nullptr;
    delete[] indices; indices = nullptr;

    context->Map(m_CBTransform, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    {
        auto rawdata = (CBTransformType *)mapped.pData;
        D3DXMATRIX temp;
        D3DXMatrixTranspose(&temp, &proj);
        rawdata->proj = temp;
    }
    context->Unmap(m_CBTransform, 0);

    context->Map(m_CBColor, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    {
        auto rawdata = (CBColorType *)mapped.pData;
        rawdata->color = D3DXVECTOR4(1, 0, 0, 1);
    }
    context->Unmap(m_CBColor, 0);

    UINT stride = sizeof(VBType);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, &m_VB, &stride, &offset);
    context->IASetIndexBuffer(m_IB, DXGI_FORMAT_R32_UINT, 0);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->IASetInputLayout(m_Layout);

    context->VSSetConstantBuffers(0, 1, &m_CBTransform);
    context->VSSetShader(m_VS, nullptr, 0);

    context->PSSetConstantBuffers(0, 1, &m_CBColor);
    context->PSSetShaderResources(0, 1, &m_SRVTexture);
    context->PSSetSamplers(0, 1, &m_SamplerState);
    context->PSSetShader(m_PS, nullptr, 0);

    context->DrawIndexed(VN, 0, 0);
}

void Quin::System::DX11::Font::InitializeBuffer(ID3D11Device * device)
{
    HRESULT hr = S_OK;

    D3D11_BUFFER_DESC bufferDesc;
    D3D11_SAMPLER_DESC sampleDesc;

    /* =====VB&IB===== */
    /**
    Byte width is not static, initialize in render
    **/

    /* =====CB===== */
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.ByteWidth = sizeof(CBTransformType);
    bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 0;
    hr = device->CreateBuffer(&bufferDesc, nullptr, &m_CBTransform);
    FAILTHROW;

    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.ByteWidth = sizeof(CBColorType);
    bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 0;
    hr = device->CreateBuffer(&bufferDesc, nullptr, &m_CBColor);
    FAILTHROW;

    /* =====Texture===== */
    hr = D3DX11CreateShaderResourceViewFromFile(device, "Res/font.dds", nullptr, nullptr, &m_SRVTexture, nullptr);
    FAILTHROW;

    /* =====SamplerState===== */
    sampleDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampleDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampleDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampleDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampleDesc.MipLODBias = 0.0f;
    sampleDesc.MaxAnisotropy = 1;
    sampleDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    sampleDesc.BorderColor[0] = 0;
    sampleDesc.BorderColor[1] = 0;
    sampleDesc.BorderColor[2] = 0;
    sampleDesc.BorderColor[3] = 0;
    sampleDesc.MinLOD = 0;
    sampleDesc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = device->CreateSamplerState(&sampleDesc, &m_SamplerState);
    FAILTHROW;
}

void Quin::System::DX11::Font::ShutdownBuffer()
{
    SafeRelease(&m_SamplerState);
    SafeRelease(&m_CBTransform);
    SafeRelease(&m_IB);
    SafeRelease(&m_VB);
}

void Quin::System::DX11::Font::InitializeShader(ID3D11Device * device)
{
    HRESULT hr = S_OK;

    ID3D10Blob *blob = nullptr;
    const UINT nLayout = 2;
    D3D11_INPUT_ELEMENT_DESC layout[nLayout];

    UINT shaderFlag = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;

    SafeRelease(&blob);
    hr = D3DX11CompileFromFile("Res/font_vs.hlsl", nullptr, nullptr, "VS", "vs_5_0", shaderFlag, 0, nullptr, &blob, nullptr, nullptr);
    FAILTHROW;
    hr = device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_VS);
    FAILTHROW;

    ZeroMemory(layout, sizeof(layout));
    layout[0].SemanticName = "POSITION";
    layout[0].SemanticIndex = 0;
    layout[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
    layout[0].InputSlot = 0;
    layout[0].AlignedByteOffset = 0;
    layout[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
    layout[0].InstanceDataStepRate = 0;
    layout[1].SemanticName = "TEXCOORD";
    layout[1].SemanticIndex = 0;
    layout[1].Format = DXGI_FORMAT_R32G32_FLOAT;
    layout[1].InputSlot = 0;
    layout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
    layout[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
    layout[1].InstanceDataStepRate = 0;
    hr = device->CreateInputLayout(layout, nLayout, blob->GetBufferPointer(), blob->GetBufferSize(), &m_Layout);
    FAILTHROW;

    SafeRelease(&blob);
    hr = D3DX11CompileFromFile("Res/font_ps.hlsl", nullptr, nullptr, "PS", "ps_5_0", shaderFlag, 0, nullptr, &blob, nullptr, nullptr);
    FAILTHROW;
    hr = device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_PS);
    FAILTHROW;
    SafeRelease(&blob);
}

void Quin::System::DX11::Font::ShutdownShader()
{
    SafeRelease(&m_PS);
    SafeRelease(&m_Layout);
    SafeRelease(&m_VS);
}
