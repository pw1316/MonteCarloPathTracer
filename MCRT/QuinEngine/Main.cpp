#include <stdafx.h>

#include <Core/Application.hpp>
#include <DX11/GraphicsDX11.hpp>

int APIENTRY WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    Quin::Core::Application app(1280, 720);
    Quin::System::DX11::GraphicsDX11 g;
    app.Run(hInstance, nCmdShow, g);
}
