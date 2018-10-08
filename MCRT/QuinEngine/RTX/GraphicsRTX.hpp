#pragma once
#include <stdafx.h>

#include <Core/Graphics.hpp>

namespace Quin::RTX
{
    class GraphicsRTX : public Core::Graphics
    {
    public:
        void Initialize(HWND hWnd, UINT w, UINT h) override
        {
            DoInitialize(hWnd, w, h);
        }
        void Shutdown() override
        {
            DoShutdown();
        }
        BOOL OnUpdate() override
        {
            return DoOnUpdate();
        }
        LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) override
        {
            return DoMessageHandler(hWnd, message, wParam, lParam);
        }
    private:
        /* Override impl */
        void DoInitialize(HWND hWnd, UINT w, UINT h);
        void DoShutdown();
        BOOL DoOnUpdate();
        LRESULT CALLBACK DoMessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

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
