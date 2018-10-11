#pragma once
#include "targetver.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <assert.h>

#pragma warning(push)
#pragma warning(disable : 4005)
#include <DXGI.h>
#include <D3D11.h>
#include <D3DX11async.h>
#include <D3DX11tex.h>
#include <D3DX10math.h>
#include <D3Dcompiler.h>
#pragma warning(pop)

#include <tiny_obj_loader.h>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")

#if _DEBUG
#pragma comment(lib, "d3dx10d.lib")
#pragma comment(lib, "d3dx11d.lib")
#pragma comment(lib, "tinyobjloaderd.lib")
#else
#pragma comment(lib, "d3dx10.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "tinyobjloader.lib")
#endif

#define FAILTHROW if(FAILED(hr)) {throw 1;}

template<class T>
inline void SafeRelease(T** pp)
{
    if (*pp)
    {
        (*pp)->Release();
        *pp = nullptr;
    }
}
