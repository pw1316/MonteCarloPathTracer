#include <stdafx.h>
#include "GraphicsRTX.hpp"

#include <random>

#include <Utils/Structure.hpp>
#include <Utils/KDTree.hpp>
#include <RTX/ShaderResource.hpp>

void Quin::RTX::GraphicsRTX::DoInitialize(HWND hWnd, UINT w, UINT h)
{
    HRESULT hr = S_OK;

    ID3D11Texture2D *texture2D = nullptr;

    D3D11_TEXTURE2D_DESC texture2DDesc;
    DXGI_SWAP_CHAIN_DESC SCDesc;
    D3D11_RASTERIZER_DESC RSDesc;
    D3D11_DEPTH_STENCIL_DESC DSDesc;
    D3D11_DEPTH_STENCIL_VIEW_DESC DSVDesc;
    D3D11_VIEWPORT viewport;

    m_w = w;
    m_h = h;

    ZeroMemory(&SCDesc, sizeof(SCDesc));
    SCDesc.BufferDesc.Width = w;
    SCDesc.BufferDesc.Height = h;
    SCDesc.BufferDesc.RefreshRate.Numerator = 60;// 60FPS
    SCDesc.BufferDesc.RefreshRate.Denominator = 1;// 60FPS
    SCDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    SCDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    SCDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    SCDesc.SampleDesc.Count = 1;// MSAA off
    SCDesc.SampleDesc.Quality = 0;// MSAA off
    SCDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    SCDesc.BufferCount = 1;// Single Buffer
    SCDesc.OutputWindow = hWnd;
    SCDesc.Windowed = true;
    SCDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    SCDesc.Flags = 0;// No Advanced Flags
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    hr = D3D11CreateDeviceAndSwapChain(
        nullptr,// Adapter
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_DEBUG,
        &featureLevel, 1,
        D3D11_SDK_VERSION,
        &SCDesc, &m_swapchain,
        &m_device,
        nullptr,
        &m_context
    );
    FAILTHROW;

    /* IA */

    /* VS */

    /* Rasterizer */
    ZeroMemory(&RSDesc, sizeof(RSDesc));
    RSDesc.FillMode = D3D11_FILL_SOLID;
    RSDesc.CullMode = D3D11_CULL_BACK;
    RSDesc.FrontCounterClockwise = false;
    RSDesc.DepthBias = 0;
    RSDesc.DepthBiasClamp = 0.0f;
    RSDesc.SlopeScaledDepthBias = 0.0f;
    RSDesc.DepthClipEnable = true;
    RSDesc.ScissorEnable = false;
    RSDesc.MultisampleEnable = false;
    RSDesc.AntialiasedLineEnable = false;
    hr = m_device->CreateRasterizerState(&RSDesc, &m_rasterizerState);
    FAILTHROW;
    ZeroMemory(&viewport, sizeof(viewport));
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = static_cast<FLOAT>(w);
    viewport.Height = static_cast<FLOAT>(h);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    /* PS */

    /* OM */
    hr = m_swapchain->GetBuffer(0, IID_PPV_ARGS(&texture2D));
    FAILTHROW;
    hr = m_device->CreateRenderTargetView(texture2D, nullptr, &m_RTV);
    FAILTHROW;
    SafeRelease(&texture2D);
    ZeroMemory(&texture2DDesc, sizeof(texture2DDesc));
    texture2DDesc.Width = w;
    texture2DDesc.Height = h;
    texture2DDesc.MipLevels = 1;
    texture2DDesc.ArraySize = 1;
    texture2DDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    texture2DDesc.SampleDesc.Count = 1;
    texture2DDesc.SampleDesc.Quality = 0;
    texture2DDesc.Usage = D3D11_USAGE_DEFAULT;
    texture2DDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    texture2DDesc.CPUAccessFlags = 0;
    texture2DDesc.MiscFlags = 0;
    hr = m_device->CreateTexture2D(&texture2DDesc, NULL, &texture2D);
    FAILTHROW;
    ZeroMemory(&DSVDesc, sizeof(DSVDesc));
    DSVDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    DSVDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    DSVDesc.Texture2D.MipSlice = 0;
    hr = m_device->CreateDepthStencilView(texture2D, &DSVDesc, &m_DSV);
    FAILTHROW;
    SafeRelease(&texture2D);
    ZeroMemory(&DSDesc, sizeof(DSDesc));
    DSDesc.DepthEnable = true;
    DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    DSDesc.DepthFunc = D3D11_COMPARISON_LESS;
    DSDesc.StencilEnable = false;
    DSDesc.StencilReadMask = 0xFF;
    DSDesc.StencilWriteMask = 0xFF;
    DSDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
    DSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    DSDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
    DSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    hr = m_device->CreateDepthStencilState(&DSDesc, &m_DSS);
    FAILTHROW;
}

void Quin::RTX::GraphicsRTX::DoShutdown()
{
    m_context->OMSetRenderTargets(0, nullptr, nullptr);
    m_context->OMSetDepthStencilState(nullptr, 1);

    m_context->RSSetViewports(0, nullptr);
    m_context->RSSetState(nullptr);

    SafeRelease(&m_DSS);
    SafeRelease(&m_DSV);
    SafeRelease(&m_RTV);

    SafeRelease(&m_rasterizerState);

    SafeRelease(&m_context);
    SafeRelease(&m_device);
    SafeRelease(&m_swapchain);
}

BOOL Quin::RTX::GraphicsRTX::DoOnUpdate()
{
    static Utils::Model model("Res/scene01.obj", "Res/");
    static Utils::KDTree tree(model.attr, model.shapes);
    static ShaderResource SR(m_device, model, m_w, m_h);

    D3DXMATRIX viewMatrix;
    {
        D3DXVECTOR3 veye(0, 5, 17);
        D3DXVECTOR3 vat(0, 5, 16);
        D3DXVECTOR3 vup(0, 1, 0);
        D3DXMatrixLookAtRH(&viewMatrix, &veye, &vat, &vup);
        D3DXMatrixTranspose(&viewMatrix, &viewMatrix);
    }
    FLOAT color[] = { 0.2f, 0.15f, 0.15f, 0.0f };
    m_context->ClearRenderTargetView(m_RTV, color);
    m_context->ClearDepthStencilView(m_DSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
    
    std::mt19937 rng(1234);
    //TODO Dispatch

    m_swapchain->Present(1, 0);
    return true;
}

LRESULT Quin::RTX::GraphicsRTX::DoMessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    return DefWindowProc(hWnd, message, wParam, lParam);
}
