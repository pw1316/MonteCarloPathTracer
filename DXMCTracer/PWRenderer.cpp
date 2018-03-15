#include "PWRenderer.hpp"

#include <algorithm>
#include <cstdio>
#include <omp.h>

#define PI 3.1415926535898

/* Static Menber */
PWGL* PWGL::instance_ = nullptr;
LPCWSTR PWGL::WINDOW_CLASS_NAME = L"PWGL";
LPCWSTR PWGL::WINDOW_NAME = L"MONTE CARLO PATH TRACER";
const PWint PWGL::WINDOW_WIDTH = 800;
const PWint PWGL::WINDOW_HEIGHT = 600;

/* Methods */
PWGL *PWGL::getInstance()
{
    if (PWGL::instance_ == nullptr)
    {
        instance_ = new PWGL();
    }
    return PWGL::instance_;
}

void PWGL::releaseInstance()
{
    if (PWGL::instance_ != nullptr)
    {
        delete PWGL::instance_;
        PWGL::instance_ = nullptr;
    }
}

HRESULT PWGL::initWindow()
{
    HRESULT hr = S_OK;
    if (SUCCEEDED(hr))
    {
        WNDCLASSEX wcex = {};
        wcex.cbSize = sizeof(WNDCLASSEX);
        wcex.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = PWGL::wndProc;
        wcex.cbClsExtra = 0;
        wcex.cbWndExtra = sizeof(LONG_PTR);
        wcex.hInstance = HINST_THISCOMPONENT;
        wcex.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
        wcex.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        wcex.lpszMenuName = NULL;
        wcex.lpszClassName = PWGL::WINDOW_CLASS_NAME;
        wcex.hIconSm = LoadIcon(HINST_THISCOMPONENT, IDI_APPLICATION);
        RegisterClassEx(&wcex);
    }
    hWND_ = CreateWindowEx(
        0,
        PWGL::WINDOW_CLASS_NAME,
        PWGL::WINDOW_NAME,
        WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME ^ WS_MAXIMIZEBOX,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        PWGL::WINDOW_WIDTH,
        PWGL::WINDOW_HEIGHT,
        NULL,
        NULL,
        HINST_THISCOMPONENT,
        this
    );
    hr = hWND_ ? S_OK : E_FAIL;
    if (SUCCEEDED(hr))
    {
        hr = initDevice();
    }
    if (SUCCEEDED(hr))
    {
        ShowWindow(hWND_, SW_SHOWNORMAL);
        UpdateWindow(hWND_);
    }
    return hr;
}

HRESULT PWGL::initDevice()
{
    HRESULT hr = S_OK;

    IDXGIFactory *factory = nullptr;
    IDXGIAdapter *adapter = nullptr;
    DXGI_ADAPTER_DESC adapterDesc;
    UINT numModes = 0;
    hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    hr = factory->EnumAdapters(0, &adapter);
    adapter->GetDesc(&adapterDesc);
    SafeRelease(&factory);
    SafeRelease(&adapter);

    RECT rc;
    UINT bufferH, bufferW;
    GetClientRect(hWND_, &rc);
    bufferW = rc.right - rc.left;
    bufferH = rc.bottom - rc.top;

    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    swapChainDesc.BufferDesc.Width = bufferW;
    swapChainDesc.BufferDesc.Height = bufferH;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    /* No MSAA */
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 1;
    swapChainDesc.OutputWindow = hWND_;
    swapChainDesc.Windowed = true;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swapChainDesc.Flags = 0;

    D3D_FEATURE_LEVEL featureLevel;
    hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_SINGLETHREADED,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &swapChainDesc,
        &m_swapChain,
        &m_d3dDevice,
        &featureLevel,
        &m_d3dContext
    );

    ID3D11Texture2D *backBuffer = nullptr;
    m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    m_d3dDevice->CreateRenderTargetView(backBuffer, 0, &m_RTV);
    SafeRelease(&backBuffer);

    D3D11_TEXTURE2D_DESC depthStencilBufferDesc;
    depthStencilBufferDesc.Width = bufferW;
    depthStencilBufferDesc.Height = bufferH;
    depthStencilBufferDesc.MipLevels = 1;
    depthStencilBufferDesc.ArraySize = 1;
    depthStencilBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthStencilBufferDesc.SampleDesc.Count = 1;
    depthStencilBufferDesc.SampleDesc.Quality = 0;
    depthStencilBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    depthStencilBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    depthStencilBufferDesc.CPUAccessFlags = 0;
    depthStencilBufferDesc.MiscFlags = 0;
    m_d3dDevice->CreateTexture2D(&depthStencilBufferDesc, 0, &m_dsBuffer);

    D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
    ZeroMemory(&depthStencilDesc, sizeof(depthStencilDesc));
    depthStencilDesc.DepthEnable = true;
    depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS;
    depthStencilDesc.StencilEnable = true;
    depthStencilDesc.StencilReadMask = 0xff;
    depthStencilDesc.StencilWriteMask = 0xff;
    depthStencilDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
    depthStencilDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    depthStencilDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
    depthStencilDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    depthStencilDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    m_d3dDevice->CreateDepthStencilState(&depthStencilDesc, &m_DSS);
    m_d3dContext->OMSetDepthStencilState(m_DSS, 1);

    m_d3dDevice->CreateDepthStencilView(m_dsBuffer, 0, &m_DSV);
    m_d3dContext->OMSetRenderTargets(1, &m_RTV, m_DSV);

    D3D11_VIEWPORT viewPort;
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = static_cast<FLOAT>(bufferW);
    viewPort.Height = static_cast<FLOAT>(bufferH);
    viewPort.MinDepth = 0;
    viewPort.MaxDepth = 1;
    m_d3dContext->RSSetViewports(1, &viewPort);

    ///* Resource */
    //Vertex vertices[] =
    //{
    //    { -1,-1,-1,1,1,1,1 },
    //    { -1,+1,-1,0,0,0,1 },
    //    { +1,+1,-1,1,0,0,1 },
    //    { +1,-1,-1,0,1,0,1 },
    //    { -1,-1,+1,0,0,1,1 },
    //    { -1,+1,+1,1,1,0,1 },
    //    { +1,+1,+1,1,0,1,1 },
    //    { +1,-1,+1,0,1,1,1 }
    //};
    //D3D11_BUFFER_DESC bdesc;
    //bdesc.Usage = D3D11_USAGE_IMMUTABLE;
    //bdesc.ByteWidth = sizeof(Vertex) * 8;
    //bdesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    //bdesc.CPUAccessFlags = 0;
    //bdesc.MiscFlags = 0;
    //bdesc.StructureByteStride = 0;
    //D3D11_SUBRESOURCE_DATA vinit;
    //vinit.pSysMem = vertices;
    //m_d3dDevice->CreateBuffer(&bdesc, &vinit, &VB);

    //UINT indices[] =
    //{
    //    0,1,2,
    //    0,2,3,
    //    4,6,5,
    //    4,7,6,
    //    4,5,1,
    //    4,1,0,
    //    3,2,6,
    //    3,6,7,
    //    1,5,6,
    //    1,6,2,
    //    4,0,3,
    //    4,3,7
    //};
    //bdesc.Usage = D3D11_USAGE_IMMUTABLE;
    //bdesc.ByteWidth = sizeof(UINT) * 36;
    //bdesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    //bdesc.CPUAccessFlags = 0;
    //bdesc.MiscFlags = 0;
    //bdesc.StructureByteStride = 0;
    //vinit.pSysMem = indices;
    //m_d3dDevice->CreateBuffer(&bdesc, &vinit, &IB);

    //ID3D10Blob *vshader = nullptr;
    //hr = D3DX11CompileFromFile(L"VS.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", 0, 0, nullptr, &vshader, nullptr, nullptr);
    //m_d3dDevice->CreateVertexShader(vshader->GetBufferPointer(), vshader->GetBufferSize(), nullptr, &VS);
    //ID3D10Blob *pshader = nullptr;
    //hr = D3DX11CompileFromFile(L"PS.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", 0, 0, nullptr, &pshader, nullptr, nullptr);
    //m_d3dDevice->CreatePixelShader(pshader->GetBufferPointer(), pshader->GetBufferSize(), nullptr, &PS);

    //D3D11_INPUT_ELEMENT_DESC layout[] =
    //{
    //    { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    //    { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    //};
    //m_d3dDevice->CreateInputLayout(layout, 2, vshader->GetBufferPointer(), vshader->GetBufferSize(), &vLayout);
    //SafeRelease(&vshader);
    //SafeRelease(&pshader);

    //D3D11_BUFFER_DESC cbdesc;
    //cbdesc.Usage = D3D11_USAGE_DYNAMIC;
    //cbdesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    //cbdesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    //cbdesc.MiscFlags = 0;
    //cbdesc.ByteWidth = sizeof(VSPerObject);
    //m_d3dDevice->CreateBuffer(&cbdesc, nullptr, &VSCB);

    /* FPS counter */
    QueryPerformanceFrequency(&frequency_);
    return hr;
}

void PWGL::mainLoop()
{
    MSG msg;
    ZeroMemory(&msg, sizeof(MSG));
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
}

HRESULT PWGL::onRender()
{
    FLOAT color[] = { 0.0f, 0.125f, 0.3f, 1.0f };
    m_d3dContext->ClearRenderTargetView(m_RTV, color);
    m_d3dContext->ClearDepthStencilView(m_DSV, D3D11_CLEAR_DEPTH, 1.0, 0);
    m_swapChain->Present(0, 0);

    //D3DXMATRIX V;
    //D3DXMATRIX P;
    //D3DXVECTOR3 eye(5, 5, 5);
    //D3DXVECTOR3 at(0, 0, 0);
    //D3DXVECTOR3 up(0, 1, 0);
    //D3DXMatrixLookAtLH(&V, &eye, &at, &up);
    //D3DXMatrixPerspectiveFovLH(&P, 60, 1.333, 1, 100);

    //m_d3dContext->IASetInputLayout(vLayout);
    //UINT stride[1];
    //UINT offset[1];
    //stride[0] = sizeof(Vertex);
    //offset[0] = 0;
    //m_d3dContext->IASetVertexBuffers(0, 1, &VB, stride, offset);
    //m_d3dContext->IASetIndexBuffer(IB, DXGI_FORMAT_R32_UINT, 0);
    //m_d3dContext->VSSetShader(VS, nullptr, 0);
    //m_d3dContext->PSSetShader(PS, nullptr, 0);
    //D3DXMATRIX mWVP = V * P;
    //D3D11_MAPPED_SUBRESOURCE mapped;
    //m_d3dContext->Map(VSCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    //VSPerObject* pVSPerObject = (VSPerObject*)mapped.pData;
    //D3DXMatrixTranspose(&pVSPerObject->mWVP, &mWVP);
    //m_d3dContext->Unmap(VSCB, 0);
    //m_d3dContext->VSSetConstantBuffers(0, 1, &VSCB);
    //m_d3dContext->DrawIndexed(36, 0, 0);

    return S_OK;
}

void PWGL::onResize(UINT width, UINT height)
{
}

LRESULT PWGL::wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    LRESULT result = 0;
    if (message == WM_CREATE)
    {
        result = 1;
    }
    else
    {
        PWGL *ppwgl = PWGL::getInstance();
        bool wasHandled = false;
        if (ppwgl)
        {
            switch (message)
            {
            case WM_PAINT:
                ppwgl->onRender();
                result = 0;
                wasHandled = true;
                break;
            case WM_SIZE:
            {
                UINT width = LOWORD(lParam);
                UINT height = HIWORD(lParam);
                ppwgl->onResize(width, height);
            }
            result = 0;
            wasHandled = true;
            break;

            case WM_DESTROY:
            {
                PostQuitMessage(0);
            }
            result = 1;
            wasHandled = true;
            break;

            case WM_MOUSEWHEEL:
            {
            }
            result = 0;
            wasHandled = true;
            break;

            case WM_KEYDOWN:
            {
                switch (wParam)
                {
                case VK_LEFT:
                    break;
                case VK_RIGHT:
                    break;
                case VK_UP:
                    break;
                case VK_DOWN:
                    break;
                }
            }
            result = 0;
            wasHandled = true;
            break;

            case WM_KEYUP:
            {
                switch (wParam)
                {
                case VK_LEFT:
                case VK_RIGHT:
                    break;
                case VK_UP:
                case VK_DOWN:
                    break;
                }
            }
            result = 0;
            wasHandled = true;
            break;
            }
        }
        if (!wasHandled)
        {
            result = DefWindowProc(hWnd, message, wParam, lParam);
        }
    }
    return result;
}
