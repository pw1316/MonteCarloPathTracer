#include <stdafx.h>
#include "Application.hpp"
#include <Core/Graphics.hpp>

void Quin::Core::Application::Run(HINSTANCE hInst, INT nCmdShow, Graphics& graphics)
{
    LoadString(hInst, IDS_APP_TITLE, m_AppTitle, MAX_LOADSTRING);
    LoadString(hInst, IDC_WINDOW_CLASS, m_WindowClass, MAX_LOADSTRING);
    InitializeWindowClass(hInst, nCmdShow);
    InitializeWindow(hInst, nCmdShow, graphics);
    graphics.Initialize(m_hWnd, m_w, m_h);

    MSG msg;
    ZeroMemory(&msg, sizeof(MSG));
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!graphics.OnUpdate())
        {
            break;
        }
    }

    graphics.Shutdown();
    ShutdownWindow();
    ShutdownWindowClass();
}

LRESULT Quin::Core::Application::WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CREATE:
    {
        LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
        break;
    }
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        break;
    }
    default:
    {
        auto pgraphics = reinterpret_cast<Graphics*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        return pgraphics ? pgraphics->MessageHandler(hWnd, message, wParam, lParam) : DefWindowProc(hWnd, message, wParam, lParam);
    }
    }
    return 0;
}

void Quin::Core::Application::InitializeWindowClass(HINSTANCE hInst, INT nCmdShow)
{
    UNREFERENCED_PARAMETER(nCmdShow);
    HRESULT hr = S_OK;
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WinProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInst;
    wcex.hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = m_WindowClass;
    wcex.hIconSm = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON_SMALL));
    hr = (RegisterClassEx(&wcex) == 0) ? E_FAIL : S_OK;
    FAILTHROW;
}
void Quin::Core::Application::ShutdownWindowClass()
{
    auto hInst = reinterpret_cast<HINSTANCE>(GetWindowLongPtr(m_hWnd, GWLP_HINSTANCE));
    UnregisterClass(m_WindowClass, hInst);
}

void Quin::Core::Application::InitializeWindow(HINSTANCE hInst, INT nCmdShow, Graphics& graphics)
{
    HRESULT hr = S_OK;
    RECT paintRect{ 0, 0, static_cast<LONG>(m_w), static_cast<LONG>(m_h) };
    hr = AdjustWindowRect(&paintRect, WS_OVERLAPPEDWINDOW, false) ? S_OK : E_FAIL;
    FAILTHROW;
    m_hWnd = CreateWindow
    (
        m_WindowClass,
        m_AppTitle,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        0,
        paintRect.right - paintRect.left,
        paintRect.bottom - paintRect.top,
        nullptr,
        nullptr,
        hInst,
        &graphics
    );
    hr = m_hWnd ? S_OK : E_FAIL;
    FAILTHROW;
    ShowWindow(m_hWnd, nCmdShow);
    UpdateWindow(m_hWnd);
}
void Quin::Core::Application::ShutdownWindow()
{
    ShowCursor(true);
    DestroyWindow(m_hWnd);
    m_hWnd = nullptr;
}
