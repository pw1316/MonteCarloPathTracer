#pragma once
#include <stdafx.h>
#include <Resource.h>

namespace Quin::Core
{
    const UINT MAX_LOADSTRING = 100;

    class Graphics;

    class Application
    {
    private:
    public:
        Application() = default;
        Application(UINT w, UINT h) :m_w(w), m_h(h) {}
        ~Application() = default;
        /* No copy, no move */
        Application(const Application &rhs) = delete;
        Application(Application &&rhs) = delete;
        Application &operator=(const Application &rhs) = delete;
        Application &operator=(Application &&rhs) = delete;

        void Run(HINSTANCE hInst, INT nCmdShow, Graphics& graphics);
    private:
        static LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
        void InitializeWindowClass(HINSTANCE hInst, INT nCmdShow);
        void ShutdownWindowClass();
        void InitializeWindow(HINSTANCE hInst, INT nCmdShow, Graphics& graphics);
        void ShutdownWindow();

        CHAR m_AppTitle[MAX_LOADSTRING];
        CHAR m_WindowClass[MAX_LOADSTRING];
        HWND m_hWnd = nullptr;
        UINT m_w = 640;
        UINT m_h = 480;
    };
}