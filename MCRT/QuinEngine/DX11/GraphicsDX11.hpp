#pragma once
#include <stdafx.h>

#include <Core/Graphics.hpp>

#pragma warning(push)
#pragma warning(disable : 4005)
#include <D3D11.h>
#include <D3DX10math.h>
#pragma comment(lib, "d3d11.lib")
#if _DEBUG
#pragma comment(lib, "d3dx10d.lib")
#else
#pragma comment(lib, "d3dx10.lib")
#endif
#pragma warning(pop)

#include <tiny_obj_loader.h>
#if _DEBUG
#pragma comment(lib, "tinyobjloaderd.lib")
#else
#pragma comment(lib, "tinyobjloader.lib")
#endif

namespace Quin::System::DX11
{
    class GraphicsDX11 : public Core::Graphics
    {
    public:
        void Initialize(HWND hWnd, UINT w, UINT h) override
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
        void Shutdown() override
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
        BOOL OnUpdate() override
        {
            BeginScene();
            EndScene();
            return true;
        }
        LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) override
        {
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
    private:
        void BeginScene()
        {
            FLOAT color[] = { 0.2f, 0.15f, 0.15f, 0.0f };
            m_context->ClearRenderTargetView(m_RTV, color);
            m_context->ClearDepthStencilView(m_DSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
        }
        void EndScene()
        {
            m_swapchain->Present(1, 0);
        }
        UINT m_w = 640U;
        UINT m_h = 480U;

        /* D3D Handle */
        IDXGISwapChain *m_swapchain = nullptr;
        ID3D11Device *m_device = nullptr;
        ID3D11DeviceContext *m_context = nullptr;

        /* Rasterizer */
        ID3D11RasterizerState *m_rasterizerState = nullptr;

        /* OM */
        ID3D11RenderTargetView *m_RTV = nullptr;
        ID3D11DepthStencilView *m_DSV = nullptr;
        ID3D11DepthStencilState *m_DSS = nullptr;
    };
}
