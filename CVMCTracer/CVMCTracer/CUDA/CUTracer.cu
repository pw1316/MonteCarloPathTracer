#include "CUTracer.h"

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#include <vector>

#include <CUDA/Utils.hpp>

#include "Framework/Geometry.h"
#include "Framework/Math.hpp"

namespace PW
{
    namespace Tracer
    {
        typedef struct _HitInfo
        {
            PWint objID;
            PWint triID;
            PWfloat beta;
            PWfloat gamma;
            PWVector3f hitPoint;
        } HitInfo;

        __device__ PWVector3f* vertexBuffer;
        __device__ PWuint nVertexBuffer;
        __device__ PWVector3f* normalBuffer;
        __device__ PWuint nNormalBuffer;
        __device__ Geometry::Triangle* triangleBuffer;
        __device__ PWuint nTriangleBuffer;
        __device__ Geometry::Geometry* geometryBuffer;
        __device__ PWuint nGeometryBuffer;

        __device__ HitInfo intersect(PWVector3f pos, PWVector3f dir)
        {
            HitInfo hitInfo;
            hitInfo.objID = -1;
            PWfloat tmin = FLT_MAX;
            for (PWuint geoId = 0; geoId < nGeometryBuffer; geoId++)
            {
                PWuint offset = geometryBuffer[geoId].startIndex;
                for (PWuint triId = 0; triId < geometryBuffer[geoId].numTriangles; triId++)
                {
                    const Geometry::Triangle &triangle = triangleBuffer[triId + offset];
                    const PWVector3f a = vertexBuffer[triangle.v[0]];
                    const PWVector3f b = vertexBuffer[triangle.v[1]];
                    const PWVector3f c = vertexBuffer[triangle.v[2]];
                    Math::Matrix33f betaM(
                        a.x - pos.x, a.x - c.x, dir.x,
                        a.y - pos.y, a.y - c.y, dir.y,
                        a.z - pos.z, a.z - c.z, dir.z
                    );
                    Math::Matrix33f gammaM(
                        a.x - b.x, a.x - pos.x, dir.x,
                        a.y - b.y, a.y - pos.y, dir.y,
                        a.z - b.z, a.z - pos.z, dir.z
                    );
                    Math::Matrix33f tM(
                        a.x - b.x, a.x - c.x, a.x - pos.x,
                        a.y - b.y, a.y - c.y, a.y - pos.y,
                        a.z - b.z, a.z - c.z, a.z - pos.z
                    );
                    Math::Matrix33f A(
                        a.x - b.x, a.x - c.x, dir.x,
                        a.y - b.y, a.y - c.y, dir.y,
                        a.z - b.z, a.z - c.z, dir.z
                    );
                    PWfloat detA = A.det();
                    PWfloat beta = betaM.det() / detA;
                    PWfloat gamma = gammaM.det() / detA;
                    PWfloat t = tM.det() / detA;
                    if (beta + gamma < 1 && beta > 0 && gamma > 0 && t > 0 && t < tmin)
                    {
                        tmin = t;
                        hitInfo.objID = geoId;
                        hitInfo.triID = triId + offset;
                        hitInfo.beta = beta;
                        hitInfo.gamma = gamma;
                        hitInfo.hitPoint.x = pos.x + t * dir.x;
                        hitInfo.hitPoint.y = pos.y + t * dir.y;
                        hitInfo.hitPoint.z = pos.z + t * dir.z;
                    }
                }
            }
            return hitInfo;
        }

        __inline__ __device__ PWVector3f sampleMC(curandState *RNG, PWVector3f pos, PWVector3f dir)
        {
            HitInfo hit;
            PWVector3f color(1, 1, 1);
            PWint depth;
            for (depth = 0; depth < 7; depth++)
            {
                hit = intersect(pos, dir);
                /* not hit */
                if (hit.objID == -1)
                {
                    color.x = 0;
                    color.y = 0;
                    color.z = 0;
                    depth = -1;
                    break;
                }
                /* hit */
                else
                {
                    const Geometry::Geometry &hitObj = geometryBuffer[hit.objID];
                    /* light */
                    if (hitObj.material.Ka.x > 0)
                    {
                        color.x *= ILLUM;
                        color.y *= ILLUM;
                        color.z *= ILLUM;
                        depth = -1;
                        break;
                    }
                    /* Non-light */
                    else
                    {
                        PWVector3f normal1 = normalBuffer[triangleBuffer[hit.triID].n[0]];
                        PWVector3f normal2 = normalBuffer[triangleBuffer[hit.triID].n[1]];
                        PWVector3f normal3 = normalBuffer[triangleBuffer[hit.triID].n[2]];
                        PWVector3f normal;
                        normal.x = normal1.x * (1.0f - hit.beta - hit.gamma) + normal2.x * hit.beta + normal3.x * hit.gamma;
                        normal.y = normal1.y * (1.0f - hit.beta - hit.gamma) + normal2.y * hit.beta + normal3.y * hit.gamma;
                        normal.z = normal1.z * (1.0f - hit.beta - hit.gamma) + normal2.z * hit.beta + normal3.z * hit.gamma;
                        CUDA::normalize(normal);
                        /* Transparent */
                        if (hitObj.material.Tr > 0)
                        {
                            dir = CUDA::sampleFresnel(RNG, normal, dir, 1, 1.0 / geometryBuffer[hit.objID].material.Ni);
                            if (dir.x == 0 && dir.y == 0 && dir.z == 0)
                            {
                                color.x = 1;
                                color.y = 1;
                                color.z = 0;
                                depth = -1;
                                break;
                            }
                            color.x *= geometryBuffer[hit.objID].material.Kd.x;
                            color.y *= geometryBuffer[hit.objID].material.Kd.y;
                            color.z *= geometryBuffer[hit.objID].material.Kd.z;
                            pos.x = hit.hitPoint.x + 0.01f * dir.x;
                            pos.y = hit.hitPoint.y + 0.01f * dir.y;
                            pos.z = hit.hitPoint.z + 0.01f * dir.z;
                        }
                        /* Specular */
                        else if (hitObj.material.Ns > 1)
                        {
                            dir = CUDA::samplePhong(RNG, normal, dir, geometryBuffer[hit.objID].material.Ns);
                            if (dir.x * normal.x + dir.y * normal.y + dir.z * normal.z < 0)
                            {
                                color.x = 0;
                                color.y = 0;
                                color.z = 0;
                                depth = -1;
                                break;
                            }
                            color.x *= geometryBuffer[hit.objID].material.Ks.x;
                            color.y *= geometryBuffer[hit.objID].material.Ks.y;
                            color.z *= geometryBuffer[hit.objID].material.Ks.z;
                            pos.x = hit.hitPoint.x + 0.01f * dir.x;
                            pos.y = hit.hitPoint.y + 0.01f * dir.y;
                            pos.z = hit.hitPoint.z + 0.01f * dir.z;
                        }
                        /* Diffuse */
                        else
                        {
                            color.x *= geometryBuffer[hit.objID].material.Kd.x;
                            color.y *= geometryBuffer[hit.objID].material.Kd.y;
                            color.z *= geometryBuffer[hit.objID].material.Kd.z;
                            if (dir.x * normal.x + dir.y * normal.y + dir.z * normal.z > 0)
                            {
                                dir = -CUDA::sampleHemi(RNG, normal);
                            }
                            else
                            {
                                dir = CUDA::sampleHemi(RNG, normal);
                            }
                            pos.x = hit.hitPoint.x + 0.01f * dir.x;
                            pos.y = hit.hitPoint.y + 0.01f * dir.y;
                            pos.z = hit.hitPoint.z + 0.01f * dir.z;
                        }
                    }
                }
            }
            if (depth != -1)
            {
                hit = intersect(pos, dir);
                if (geometryBuffer[hit.objID].material.Ka.x > 0)
                {
                    color.x *= ILLUM;
                    color.y *= ILLUM;
                    color.z *= ILLUM;
                }
                else
                {
                    color.x = 0;
                    color.y = 0;
                    color.z = 0;
                }
            }
            return color;
        }

        __global__ void rayTraceKernel(PWVector4f *c, PWuint seedOffset)
        {
            PWuint x = threadIdx.x;
            PWuint y = blockIdx.x;
            PWuint height = gridDim.x;
            PWuint width = blockDim.x;
            /* Init RNG */
            curandState RNG;
            curand_init(y * width + x + seedOffset, 0, 0, &RNG);
            /* Camera Params inline */
            PWVector3f camEye(0, 5, 17);
            PWVector3f camDir(0, 0, -1);
            PWVector3f camUp(0, 1, 0);
            PWVector3f camRight(1, 0, 0);
            /* Project Params inline */
            PWfloat projFOV = 60; // degree

            /* Reproject */
            PWVector3f initRayDir;
            initRayDir.x = (2.0 * x / width - 1) * tan(projFOV * PW_PI / 360);
            initRayDir.y = (1.0 * height / width - 2.0 * y / width) * tan(projFOV * PW_PI / 360);
            initRayDir.z = -1;
            /* View to World */
            PWVector3f worldRay;
            worldRay.x = camRight.x * initRayDir.x + camUp.x * initRayDir.y - camDir.x * initRayDir.z;
            worldRay.y = camRight.y * initRayDir.x + camUp.y * initRayDir.y - camDir.y * initRayDir.z;
            worldRay.z = camRight.z * initRayDir.x + camUp.z * initRayDir.y - camDir.z * initRayDir.z;
            PW::CUDA::normalize(worldRay);

            /* MC Sampling */
            PWVector3f color(0, 0, 0);
            for (int i = 0; i < 100; i++)
            {
                PWVector3f temp = sampleMC(&RNG, camEye, worldRay);
                color.x += temp.x;
                color.y += temp.y;
                color.z += temp.z;
            }
            c[y * width + x].x = color.x / 100;
            c[y * width + x].y = color.y / 100;
            c[y * width + x].z = color.z / 100;

            c[y * width + x].w = 0;
        }

        cudaError_t RenderScene1(const PW::FileReader::ObjModel *model, PWVector4f *hostcolor)
        {
            cudaError_t cudaStatus;
            cudaStatus = cudaSetDevice(0);
            if (cudaStatus != cudaSuccess)
            {
                return cudaStatus;
            }

            PWuint deviceVertexBufferNum = model->m_vertices.size();
            cudaMemcpyToSymbol(nVertexBuffer, &deviceVertexBufferNum, sizeof(PWuint));
            void* deviceVertexBufferAddr = nullptr;
            cudaMalloc((void**)&deviceVertexBufferAddr, sizeof(PWVector3f) * deviceVertexBufferNum);
            cudaMemcpyToSymbol(vertexBuffer, &deviceVertexBufferAddr, sizeof(void*));
            std::vector<PWVector3f> hostVertexBuffer(deviceVertexBufferNum);
            for (PWuint i = 0; i < deviceVertexBufferNum; ++i)
            {
                hostVertexBuffer[i].x = model->m_vertices[i].getX();
                hostVertexBuffer[i].y = model->m_vertices[i].getY();
                hostVertexBuffer[i].z = model->m_vertices[i].getZ();
            }
            cudaMemcpy(deviceVertexBufferAddr, &hostVertexBuffer[0], sizeof(PWVector3f) * deviceVertexBufferNum, cudaMemcpyHostToDevice);

            PWuint deviceNormalBufferNum = model->m_normals.size();
            cudaMemcpyToSymbol(nNormalBuffer, &deviceNormalBufferNum, sizeof(PWuint));
            void* deviceNormalBufferAddr = nullptr;
            cudaMalloc((void**)&deviceNormalBufferAddr, sizeof(PWVector3f) * deviceNormalBufferNum);
            cudaMemcpyToSymbol(normalBuffer, &deviceNormalBufferAddr, sizeof(void*));
            std::vector<PWVector3f> hostNormalBuffer(deviceNormalBufferNum);
            for (PWuint i = 0; i < deviceNormalBufferNum; ++i)
            {
                hostNormalBuffer[i].x = model->m_normals[i].getX();
                hostNormalBuffer[i].y = model->m_normals[i].getY();
                hostNormalBuffer[i].z = model->m_normals[i].getZ();
            }
            cudaMemcpy(deviceNormalBufferAddr, &hostNormalBuffer[0], sizeof(PWVector3f) * deviceNormalBufferNum, cudaMemcpyHostToDevice);

            PWuint deviceTriangleBufferNum = model->m_triangles.size();
            cudaMemcpyToSymbol(nTriangleBuffer, &deviceTriangleBufferNum, sizeof(PWuint));
            void* deviceTriangleBufferAddr = nullptr;
            cudaMalloc((void**)&deviceTriangleBufferAddr, sizeof(Geometry::Triangle) * deviceTriangleBufferNum);
            cudaMemcpyToSymbol(triangleBuffer, &deviceTriangleBufferAddr, sizeof(void*));
            std::vector<Geometry::Triangle> hostTriangleBuffer(deviceTriangleBufferNum);
            for (PWuint i = 0; i < deviceTriangleBufferNum; ++i)
            {
                hostTriangleBuffer[i].v[0] = model->m_triangles[i].m_vertexIndex[0];
                hostTriangleBuffer[i].v[1] = model->m_triangles[i].m_vertexIndex[1];
                hostTriangleBuffer[i].v[2] = model->m_triangles[i].m_vertexIndex[2];
                hostTriangleBuffer[i].n[0] = model->m_triangles[i].m_normalIndex[0];
                hostTriangleBuffer[i].n[1] = model->m_triangles[i].m_normalIndex[1];
                hostTriangleBuffer[i].n[2] = model->m_triangles[i].m_normalIndex[2];
            }
            cudaMemcpy(deviceTriangleBufferAddr, &hostTriangleBuffer[0], sizeof(Geometry::Triangle) * deviceTriangleBufferNum, cudaMemcpyHostToDevice);

            PWuint deviceGeometryBufferNum = 0;
            for (auto &group : model->m_groups)
            {
                if (group.second.m_triangleIndices.size() != 0)
                {
                    deviceGeometryBufferNum += 1;
                }
            }
            cudaMemcpyToSymbol(nGeometryBuffer, &deviceGeometryBufferNum, sizeof(PWuint));
            void* deviceGeometryBufferAddr = nullptr;
            cudaMalloc((void**)&deviceGeometryBufferAddr, sizeof(Geometry::Geometry) * deviceGeometryBufferNum);
            cudaMemcpyToSymbol(geometryBuffer, &deviceGeometryBufferAddr, sizeof(void*));
            std::vector<Geometry::Geometry> hostGeometryBuffer(deviceGeometryBufferNum);
            PWuint gIndex = 0;
            for (auto &group : model->m_groups)
            {
                if (group.second.m_triangleIndices.size() == 0)
                {
                    continue;
                }
                auto &triB = model->m_triangles;
                auto &matB = model->m_materials;
                hostGeometryBuffer[gIndex].startIndex = group.second.m_triangleIndices[0];
                hostGeometryBuffer[gIndex].numTriangles = group.second.m_triangleIndices.size();
                auto &mat = matB[triB[group.second.m_triangleIndices[0]].materialIndex];
                hostGeometryBuffer[gIndex].material.Ka = PWVector3f(mat.Ka.getX(), mat.Ka.getY(), mat.Ka.getZ());
                hostGeometryBuffer[gIndex].material.Kd = PWVector3f(mat.Kd.getX(), mat.Kd.getY(), mat.Kd.getZ());
                hostGeometryBuffer[gIndex].material.Ks = PWVector3f(mat.Ks.getX(), mat.Ks.getY(), mat.Ks.getZ());
                hostGeometryBuffer[gIndex].material.Ns = mat.Ns;
                hostGeometryBuffer[gIndex].material.Tr = mat.Tr;
                hostGeometryBuffer[gIndex].material.Ni = mat.Ni;
                gIndex += 1;
            }
            cudaMemcpy(deviceGeometryBufferAddr, &hostGeometryBuffer[0], sizeof(Geometry::Geometry) * deviceGeometryBufferNum, cudaMemcpyHostToDevice);

            PWVector4f *color = nullptr;
            cudaStatus = cudaMalloc((void**)&color, IMG_WIDTH * IMG_HEIGHT * sizeof(PWVector4f));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                cudaFree(color);
                return cudaStatus;
            }
            /* IMG_HEIGHT blocks, with IMG_WIDTH thread each block */
            rayTraceKernel << <IMG_HEIGHT, IMG_WIDTH >> > (color, 0);
            cudaDeviceSynchronize();
            cudaMemcpy(hostcolor, color, IMG_WIDTH * IMG_HEIGHT * sizeof(PWVector4f), cudaMemcpyDeviceToHost);

            return cudaStatus;
        }
    }
}
