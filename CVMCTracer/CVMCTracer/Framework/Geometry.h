#pragma once
#include "stdafx.h"

namespace PW
{
    namespace Geometry
    {
#pragma pack(push, 1)
        typedef struct _Index
        {
            PWuint v, n;
        } Index;

        typedef struct _Material
        {
            PWVector3f Ka;
            PWVector3f Kd;
            PWVector3f Ks;
            PWfloat Ns;
            PWfloat Tr;
            PWfloat Ni;
        } Material;

        typedef struct _Triangle
        {
            PWuint v[3];
            PWuint n[3];
        } Triangle;

        typedef struct _Geometry
        {
            Material material;
            PWuint startIndex;
            PWuint numTriangles;
        } Geometry;

        typedef struct _GeoHeader
        {
            PWuint type;
            Material material;
        } GeoHeader;

        typedef struct _GeoPlane
        {
            GeoHeader header;
            PWVector3f normal;
            PWfloat d;
        } GeoPlane;

        typedef struct _GeoSphere
        {
            GeoHeader header;
            PWVector3f center;
            PWfloat radius;
        } GeoSphere;
#pragma pack(pop)
    }
}
