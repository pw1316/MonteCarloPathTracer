#pragma once
#include <stdafx.h>

namespace Quin::Core
{
    class Graphics
    {
    public:
        Graphics() = default;
        virtual ~Graphics() = default;
        /* No copy, no move */
        Graphics(const Graphics& rhs) = delete;
        Graphics(Graphics&& rhs) = delete;
        Graphics &operator=(const Graphics &rhs) = delete;
        Graphics &operator=(Graphics &&rhs) = delete;

        virtual void Initialize(HWND hwnd, UINT w, UINT h) = 0;
        virtual void Shutdown() = 0;
        virtual BOOL OnUpdate() = 0;
        virtual LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) = 0;
    };
}