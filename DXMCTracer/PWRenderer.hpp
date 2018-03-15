#pragma once
#include "stdafx.h"
#include <d3d11.h>
#include <D3DX11.h>
#include <D3DX10math.h>
#include "Math.hpp"

/* Release for COM Components */
template<class Interface>
inline void SafeRelease(Interface **ppInterfaceToRelease)
{
    if (*ppInterfaceToRelease != NULL)
    {
        (*ppInterfaceToRelease)->Release();

        (*ppInterfaceToRelease) = NULL;
    }
}

/* Current hInstance */
#ifndef HINST_THISCOMPONENT
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#define HINST_THISCOMPONENT ((HINSTANCE)&__ImageBase)
#endif

class PWGL
{
public:
    static PWGL *getInstance();
    static void releaseInstance();
    HRESULT initWindow();
    HRESULT initDevice();
    void mainLoop();

private:
    PWGL() {}
    ~PWGL() {}
    HRESULT onRender();
    void onResize(UINT width, UINT height);
    static LRESULT CALLBACK wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

private:
    static PWGL *instance_;
    static LPCWSTR WINDOW_CLASS_NAME;
    static LPCWSTR WINDOW_NAME;
    const static INT WINDOW_WIDTH;
    const static INT WINDOW_HEIGHT;

    HWND hWND_ = nullptr;
    ID3D11Device *m_d3dDevice = nullptr;
    ID3D11DeviceContext *m_d3dContext = nullptr;
    IDXGISwapChain *m_swapChain = nullptr;
    ID3D11Texture2D *m_dsBuffer = nullptr;
    ID3D11RenderTargetView *m_RTV = nullptr;
    ID3D11DepthStencilView *m_DSV = nullptr;
    ID3D11DepthStencilState *m_DSS = nullptr;

    ID3D11InputLayout *vLayout = nullptr;
    ID3D11Buffer *VB = nullptr;
    ID3D11Buffer *IB = nullptr;
    ID3D11VertexShader *VS = nullptr;
    ID3D11PixelShader *PS = nullptr;
    ID3D11Buffer *VSCB = nullptr;
    struct Vertex
    {
        float x, y, z, r, g, b, a;
    };
    struct VSPerObject
    {
        D3DXMATRIX mWVP;
    };

    /* FPS counter */
    LARGE_INTEGER frequency_ = {};
    FLOAT fps_ = 60.0f;
};