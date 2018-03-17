#pragma once
#include "stdafx.h"

namespace PW
{
    namespace Geometry
    {
        typedef enum _GeoType
        {
            GEOMETRY_PLANE = 0,
            GEOMETRY_SPHERE = 1
        } GeoType;
#pragma pack(push, 1)
        typedef struct _Material
        {
            PWVector3f Ka;
            PWVector3f Kd;
            PWVector3f Ks;
            PWfloat Ns;
            PWfloat Tr;
            PWfloat Ni;
        } Material;

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

        typedef union _Geometry
        {
            GeoHeader header;
            GeoPlane plane;
            GeoSphere sphere;
        } Geometry;
#pragma pack(pop)
    }
}
