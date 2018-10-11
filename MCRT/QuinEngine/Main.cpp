#include <stdafx.h>

#include <Core/Application.hpp>
#include <RTX/GraphicsRTX.hpp>

int APIENTRY WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    Quin::Core::Application app(640, 480);
    Quin::RTX::GraphicsRTX g;
    app.Run(hInstance, nCmdShow, g);
    return 0;
}
