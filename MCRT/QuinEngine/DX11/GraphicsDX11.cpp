#include <stdafx.h>
#include "GraphicsDX11.hpp"

void Quin::System::DX11::GraphicsDX11::Initialize(HWND hWnd, UINT w, UINT h)
{
    m_width = w;
    m_height = h;
    InitializeDevice(hWnd);
    InitializeOM();
    InitializeRasterizer();

    float fov, aspect;
    fov = static_cast<FLOAT>(D3DX_PI) / 3.0f;
    aspect = static_cast<FLOAT>(w) / static_cast<FLOAT>(h);
    D3DXMatrixPerspectiveFovLH(&m_MatrixProj, fov, aspect, 0.1f, 100.0f);
    D3DXMatrixOrthoLH(&m_MatrixOrtho, static_cast<FLOAT>(w), static_cast<FLOAT>(h), 0.1f, 100.0f);

    m_model = new ModelDX11("Res/sphere");
    m_model->Initialize(m_device);
    m_gui = new Font;
    m_gui->Initialize(m_device);

    m_camera.pos = D3DXVECTOR3(0.0f, 2.0f, -10.0f);
    m_light.m_dir = D3DXVECTOR3(0.0f, 0.0f, 1.0f);
}
void Quin::System::DX11::GraphicsDX11::Shutdown()
{
    assert(m_model);
    m_model->Shutdown();
    delete m_model;
    m_model = nullptr;
    assert(m_gui);
    m_gui->Shutdown();
    delete m_gui;
    m_gui = nullptr;
    ShutdownRasterizer();
    ShutdownOM();
    ShutdownDevice();
}
BOOL Quin::System::DX11::GraphicsDX11::OnUpdate()
{
    BeginScene();
    /* 1 Physics */
    m_model->Rotate();
    /* 2 Input */
    /* 3 Script */
    /* 4 Graphics */
    OnRender();
    /* 5 GUI */
    OnGUI();
    EndScene();
    return true;
}

LRESULT Quin::System::DX11::GraphicsDX11::MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    return DefWindowProc(hWnd, message, wParam, lParam);
}

void Quin::System::DX11::GraphicsDX11::InitializeDevice(HWND hWnd)
{
    HRESULT hr = S_OK;
    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
    swapChainDesc.BufferDesc.Width = m_width;
    swapChainDesc.BufferDesc.Height = m_height;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;// 60FPS
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;// 60FPS
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    swapChainDesc.SampleDesc.Count = 1;// MSAA off
    swapChainDesc.SampleDesc.Quality = 0;// MSAA off
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 1;// Single Buffer
    swapChainDesc.OutputWindow = hWnd;
    swapChainDesc.Windowed = true;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swapChainDesc.Flags = 0;// No Advanced Flags
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    hr = D3D11CreateDeviceAndSwapChain(
        nullptr,// Adapter
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_DEBUG,
        &featureLevel, 1,
        D3D11_SDK_VERSION,
        &swapChainDesc, &m_swapChain,
        &m_device,
        nullptr,
        &m_context
    );
    FAILTHROW;
}
void Quin::System::DX11::GraphicsDX11::ShutdownDevice()
{
    SafeRelease(&m_context);
    SafeRelease(&m_device);
    SafeRelease(&m_swapChain);
}

void Quin::System::DX11::GraphicsDX11::InitializeOM()
{
    HRESULT hr = S_OK;

    ID3D11Texture2D *texture2D = nullptr;
    D3D11_TEXTURE2D_DESC texture2DDesc;
    D3D11_DEPTH_STENCIL_DESC DSDesc;
    D3D11_DEPTH_STENCIL_VIEW_DESC DSVDesc;
    D3D11_BLEND_DESC blendDesc;

    /* RTV */
    hr = m_swapChain->GetBuffer(0, IID_PPV_ARGS(&texture2D));
    FAILTHROW;
    hr = m_device->CreateRenderTargetView(texture2D, nullptr, &m_RTV);
    FAILTHROW;
    SafeRelease(&texture2D);

    /* DSV */
    ZeroMemory(&texture2DDesc, sizeof(texture2DDesc));
    texture2DDesc.Width = m_width;
    texture2DDesc.Height = m_height;
    texture2DDesc.MipLevels = 1;
    texture2DDesc.ArraySize = 1;
    texture2DDesc.Format = DXGI_FORMAT_D32_FLOAT;
    texture2DDesc.SampleDesc.Count = 1;
    texture2DDesc.SampleDesc.Quality = 0;
    texture2DDesc.Usage = D3D11_USAGE_DEFAULT;
    texture2DDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    texture2DDesc.CPUAccessFlags = 0;
    texture2DDesc.MiscFlags = 0;
    hr = m_device->CreateTexture2D(&texture2DDesc, NULL, &texture2D);
    FAILTHROW;
    ZeroMemory(&DSVDesc, sizeof(DSVDesc));
    DSVDesc.Format = DXGI_FORMAT_D32_FLOAT;
    DSVDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    DSVDesc.Texture2D.MipSlice = 0;
    hr = m_device->CreateDepthStencilView(texture2D, &DSVDesc, &m_DSV);
    FAILTHROW;
    SafeRelease(&texture2D);

    /* DSS */
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
    hr = m_device->CreateDepthStencilState(&DSDesc, &m_DSSWithZ);
    FAILTHROW;
    DSDesc.DepthEnable = false;
    hr = m_device->CreateDepthStencilState(&DSDesc, &m_DSSWithoutZ);
    FAILTHROW;

    /* Blend State */
    ZeroMemory(&blendDesc, sizeof(blendDesc));
    blendDesc.AlphaToCoverageEnable = false;
    blendDesc.IndependentBlendEnable = false;
    blendDesc.RenderTarget[0].BlendEnable = true;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = 0x0F;
    hr = m_device->CreateBlendState(&blendDesc, &m_BSWithBlend);
    FAILTHROW;
    blendDesc.RenderTarget[0].BlendEnable = false;
    hr = m_device->CreateBlendState(&blendDesc, &m_BSWithoutBlend);
    FAILTHROW;
}
void Quin::System::DX11::GraphicsDX11::ShutdownOM()
{
    SafeRelease(&m_BSWithoutBlend);
    SafeRelease(&m_BSWithBlend);
    SafeRelease(&m_DSSWithoutZ);
    SafeRelease(&m_DSSWithZ);
    SafeRelease(&m_DSV);
    SafeRelease(&m_RTV);
}

void Quin::System::DX11::GraphicsDX11::InitializeRasterizer()
{
    HRESULT hr = S_OK;

    ID3D11RasterizerState *RS = nullptr;
    D3D11_RASTERIZER_DESC RSDesc;
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
    hr = m_device->CreateRasterizerState(&RSDesc, &RS);
    FAILTHROW;
    m_context->RSSetState(RS);
    SafeRelease(&RS);

    D3D11_VIEWPORT viewport;
    ZeroMemory(&viewport, sizeof(viewport));
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = static_cast<FLOAT>(m_width);
    viewport.Height = static_cast<FLOAT>(m_height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    m_context->RSSetViewports(1, &viewport);
}
void Quin::System::DX11::GraphicsDX11::ShutdownRasterizer()
{
}

void Quin::System::DX11::GraphicsDX11::OnRender()
{
    D3DXMATRIX view, proj = m_MatrixProj, ortho = m_MatrixOrtho;
    float blendFactor[4] = { 0,0,0,0 };
    m_camera.Get(view);
    m_context->OMSetRenderTargets(1, &m_RTV, m_DSV);
    m_context->OMSetDepthStencilState(m_DSSWithZ, 1);
    m_context->OMSetBlendState(m_BSWithoutBlend, blendFactor, 0xFFFFFFFF);
    m_model->Render(m_context, view, proj, m_camera.Pos(), m_light.m_dir);
}
void Quin::System::DX11::GraphicsDX11::OnGUI()
{
    D3DXMATRIX ortho = m_MatrixOrtho;
    float blendFactor[4] = { 0,0,0,0 };
    m_context->OMSetDepthStencilState(m_DSSWithoutZ, 1);
    m_context->OMSetBlendState(m_BSWithBlend, blendFactor, 0xFFFFFFFF);
    m_gui->Render(m_device, m_context, "Sample Text", { 0, 100 }, m_width, m_height, ortho);
}
