#pragma once
#include "targetver.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <assert.h>

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
