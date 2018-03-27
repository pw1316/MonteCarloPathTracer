#include "CUTracer.h"

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#include <vector>

#include <CUDA/Utils.hpp>

#include "Framework/Geometry.h"
#include "Framework/Math.hpp"

#define CUDACheck(x) if((x) != cudaSuccess){return (x);}

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

        //__device__ PWVector3f camPos;
        //__device__ PWVector3f camForward;
        //__device__ PWVector3f camUp;
        //__device__ PWVector3f camRight;

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

        __inline__ __device__ PWVector3f sampleMC(curandState *RNG, PWVector3f pos, PWVector3f dir, const PWint maxDepth)
        {
            HitInfo hit;
            PWVector3f color(1, 1, 1);
            PWint depth;
            for (depth = 0; depth < maxDepth; depth++)
            {
                hit = intersect(pos, dir);
                /* Not hit */
                if (hit.objID == -1)
                {
                    return PWVector3f(0, 0, 0);
                }

                const Geometry::Geometry &hitObj = geometryBuffer[hit.objID];
                /* Hit light */
                if (hitObj.material.Ka.x > 0)
                {
                    color *= ILLUM;
                    return color;
                }

                PWVector3f normal1 = normalBuffer[triangleBuffer[hit.triID].n[0]];
                PWVector3f normal2 = normalBuffer[triangleBuffer[hit.triID].n[1]];
                PWVector3f normal3 = normalBuffer[triangleBuffer[hit.triID].n[2]];
                PWVector3f normal = normal1 * (1.0f - hit.beta - hit.gamma) + normal2 * hit.beta + normal3 * hit.gamma;
                CUDA::normalize(normal);
                /* Transparent */
                if (hitObj.material.Tr > 0)
                {
                    dir = CUDA::sampleFresnel(RNG, normal, dir, hitObj.material.Tr, hitObj.material.Ni);
                    color.x *= hitObj.material.Kd.x;
                    color.y *= hitObj.material.Kd.y;
                    color.z *= hitObj.material.Kd.z;
                    pos = hit.hitPoint + dir * 0.01f;
                }
                /* Specular */
                else if (hitObj.material.Ns > 1)
                {
                    dir = CUDA::samplePhong(RNG, normal, dir, geometryBuffer[hit.objID].material.Ns);
                    color.x *= geometryBuffer[hit.objID].material.Ks.x;
                    color.y *= geometryBuffer[hit.objID].material.Ks.y;
                    color.z *= geometryBuffer[hit.objID].material.Ks.z;
                    pos = hit.hitPoint + dir * 0.01f;
                }
                /* Diffuse */
                else
                {
                    color.x *= geometryBuffer[hit.objID].material.Kd.x;
                    color.y *= geometryBuffer[hit.objID].material.Kd.y;
                    color.z *= geometryBuffer[hit.objID].material.Kd.z;
                    if (CUDA::dot(dir, normal) > 0)
                    {
                        dir = -CUDA::sampleHemi(RNG, normal);
                    }
                    else
                    {
                        dir = CUDA::sampleHemi(RNG, normal);
                    }
                    pos = hit.hitPoint + dir * 0.01f;
                }
            }
            if (depth != -1)
            {
                hit = intersect(pos, dir);
                if (geometryBuffer[hit.objID].material.Ka.x > 0)
                {
                    color *= ILLUM;
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

        __global__ void rayTraceKernel1(PWVector3f *c, PWuint seedOffset)
        {
            PWuint x = threadIdx.x;
            PWuint y = blockIdx.x;
            PWuint height = gridDim.x;
            PWuint width = blockDim.x;
            /* Init RNG */
            curandState RNG;
            curand_init(y * width + x + seedOffset, 0, 0, &RNG);
            /* Camera Params inline */
            const PWVector3f camEye(0, 5, 17);
            const PWVector3f camDir(0, 0, -1);
            const PWVector3f camUp(0, 1, 0);
            const PWVector3f camRight(1, 0, 0);
            /* Project Params inline */
            PWfloat projFOV = 60; // degree

            /* MC Sampling */
            PWVector3f color(0, 0, 0);
            for (int i = 0; i < NUM_SAMPLES; i++)
            {
                /* Bias */
                PWfloat biasx = x + (curand_uniform(&RNG) * 2.0f - 1.0f);
                PWfloat biasy = y + (curand_uniform(&RNG) * 2.0f - 1.0f);

                /* Reproject */
                PWVector3f initRayDir
                (
                    (2.0 * biasx / width - 1) * tan(projFOV * PW_PI / 360),
                    (1.0 * height / width - 2.0 * biasy / width) * tan(projFOV * PW_PI / 360),
                    -1
                );
                /* View to World */
                PWVector3f worldRay;
                worldRay.x = camRight.x * initRayDir.x + camUp.x * initRayDir.y - camDir.x * initRayDir.z;
                worldRay.y = camRight.y * initRayDir.x + camUp.y * initRayDir.y - camDir.y * initRayDir.z;
                worldRay.z = camRight.z * initRayDir.x + camUp.z * initRayDir.y - camDir.z * initRayDir.z;
                PW::CUDA::normalize(worldRay);
                color += sampleMC(&RNG, camEye, worldRay, 7);
            }
            c[y * width + x].x = color.x / NUM_SAMPLES;
            c[y * width + x].y = color.y / NUM_SAMPLES;
            c[y * width + x].z = color.z / NUM_SAMPLES;
        }

        __global__ void rayTraceKernel2(PWVector3f *c, PWuint seedOffset)
        {
            PWuint x = threadIdx.x;
            PWuint y = blockIdx.x;
            PWuint height = gridDim.x;
            PWuint width = blockDim.x;
            /* Init RNG */
            curandState RNG;
            curand_init(y * width + x + seedOffset, 0, 0, &RNG);
            /* Camera Params inline */
            const PWVector3f camEye(0, 7, 23);
            const PWVector3f camDir(0, -0.08715574274765817355806427083747, -0.99619469809174553229501040247389);
            const PWVector3f camUp(0, 0.99619469809174553229501040247389, -0.08715574274765817355806427083747);
            const PWVector3f camRight(1, 0, 0);
            /* Project Params inline */
            PWfloat projFOV = 60; // degree

            /* MC Sampling */
            PWVector3f color(0, 0, 0);
            for (int i = 0; i < NUM_SAMPLES; i++)
            {
                /* Bias */
                PWfloat biasx = x + (curand_uniform(&RNG) * 2.0f - 1.0f);
                PWfloat biasy = y + (curand_uniform(&RNG) * 2.0f - 1.0f);

                /* Reproject */
                PWVector3f initRayDir
                (
                    (2.0 * biasx / width - 1) * tan(projFOV * PW_PI / 360),
                    (1.0 * height / width - 2.0 * biasy / width) * tan(projFOV * PW_PI / 360),
                    -1
                );
                /* View to World */
                PWVector3f worldRay;
                worldRay.x = camRight.x * initRayDir.x + camUp.x * initRayDir.y - camDir.x * initRayDir.z;
                worldRay.y = camRight.y * initRayDir.x + camUp.y * initRayDir.y - camDir.y * initRayDir.z;
                worldRay.z = camRight.z * initRayDir.x + camUp.z * initRayDir.y - camDir.z * initRayDir.z;
                PW::CUDA::normalize(worldRay);
                color += sampleMC(&RNG, camEye, worldRay, 7);
            }
            c[y * width + x].x = color.x / NUM_SAMPLES;
            c[y * width + x].y = color.y / NUM_SAMPLES;
            c[y * width + x].z = color.z / NUM_SAMPLES;
        }

        cudaError_t Initialize()
        {
            return cudaSetDevice(0);
        }

        cudaError_t CreateGeometry(const PW::FileReader::ObjModel *model)
        {
            cudaError_t err = cudaSuccess;

            /* Vertex buffer */
            PWuint deviceVertexBufferNum = static_cast<PWuint>(model->m_vertices.size());
            cudaMemcpyToSymbol(nVertexBuffer, &deviceVertexBufferNum, sizeof(PWuint));
            void* deviceVertexBufferAddr = nullptr;
            cudaMalloc((void**)&deviceVertexBufferAddr, sizeof(PWVector3f) * deviceVertexBufferNum);
            cudaMemcpyToSymbol(vertexBuffer, &deviceVertexBufferAddr, sizeof(void*));
            std::vector<PWVector3f> hostVertexBuffer(deviceVertexBufferNum);
            for (PWuint i = 0; i < deviceVertexBufferNum; ++i)
            {
                hostVertexBuffer[i].x = model->m_vertices[i].x;
                hostVertexBuffer[i].y = model->m_vertices[i].y;
                hostVertexBuffer[i].z = model->m_vertices[i].z;
            }
            cudaMemcpy(deviceVertexBufferAddr, &hostVertexBuffer[0], sizeof(PWVector3f) * deviceVertexBufferNum, cudaMemcpyHostToDevice);

            /* Normal buffer */
            PWuint deviceNormalBufferNum = static_cast<PWuint>(model->m_normals.size());
            cudaMemcpyToSymbol(nNormalBuffer, &deviceNormalBufferNum, sizeof(PWuint));
            void* deviceNormalBufferAddr = nullptr;
            cudaMalloc((void**)&deviceNormalBufferAddr, sizeof(PWVector3f) * deviceNormalBufferNum);
            cudaMemcpyToSymbol(normalBuffer, &deviceNormalBufferAddr, sizeof(void*));
            std::vector<PWVector3f> hostNormalBuffer(deviceNormalBufferNum);
            for (PWuint i = 0; i < deviceNormalBufferNum; ++i)
            {
                hostNormalBuffer[i].x = model->m_normals[i].x;
                hostNormalBuffer[i].y = model->m_normals[i].y;
                hostNormalBuffer[i].z = model->m_normals[i].z;
            }
            cudaMemcpy(deviceNormalBufferAddr, &hostNormalBuffer[0], sizeof(PWVector3f) * deviceNormalBufferNum, cudaMemcpyHostToDevice);

            /* Triangle buffer */
            PWuint deviceTriangleBufferNum = static_cast<PWuint>(model->m_triangles.size());
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

            /* Geomtry buffer */
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
                hostGeometryBuffer[gIndex].numTriangles = static_cast<PWuint>(group.second.m_triangleIndices.size());
                auto &mat = matB[triB[group.second.m_triangleIndices[0]].materialIndex];
                hostGeometryBuffer[gIndex].material.Ka = PWVector3f(mat.Ka.x, mat.Ka.y, mat.Ka.z);
                hostGeometryBuffer[gIndex].material.Kd = PWVector3f(mat.Kd.x, mat.Kd.y, mat.Kd.z);
                hostGeometryBuffer[gIndex].material.Ks = PWVector3f(mat.Ks.x, mat.Ks.y, mat.Ks.z);
                hostGeometryBuffer[gIndex].material.Ns = static_cast<PWfloat>(mat.Ns);
                hostGeometryBuffer[gIndex].material.Tr = static_cast<PWfloat>(mat.Tr);
                hostGeometryBuffer[gIndex].material.Ni = static_cast<PWfloat>(mat.Ni);
                gIndex += 1;
            }
            cudaMemcpy(deviceGeometryBufferAddr, &hostGeometryBuffer[0], sizeof(Geometry::Geometry) * deviceGeometryBufferNum, cudaMemcpyHostToDevice);
            
            return err;
        }

        cudaError_t DestroyGeometry()
        {
            cudaError_t err = cudaSuccess;

            void* deviceVertexBufferAddr = nullptr;
            void* deviceNormalBufferAddr = nullptr;
            void* deviceTriangleBufferAddr = nullptr;
            void* deviceGeometryBufferAddr = nullptr;

            cudaGetSymbolAddress(&deviceVertexBufferAddr, vertexBuffer);
            cudaFree(deviceVertexBufferAddr);

            cudaGetSymbolAddress(&deviceNormalBufferAddr, normalBuffer);
            cudaFree(deviceNormalBufferAddr);

            cudaGetSymbolAddress(&deviceTriangleBufferAddr, triangleBuffer);
            cudaFree(deviceTriangleBufferAddr);

            cudaGetSymbolAddress(&deviceGeometryBufferAddr, geometryBuffer);
            cudaFree(deviceGeometryBufferAddr);

            return err;
        }

        cudaError_t RenderScene(const PWint sceneID, PWVector3f *hostcolor)
        {
            cudaError_t err;

            PWVector3f *color = nullptr;
            err = cudaMalloc((void**)&color, IMG_WIDTH * IMG_HEIGHT * sizeof(PWVector3f));
            /* IMG_HEIGHT blocks, with IMG_WIDTH thread each block */
            if (sceneID == 1)
            {
                rayTraceKernel1 << <IMG_HEIGHT, IMG_WIDTH >> > (color, 0);
            }
            else
            {
                rayTraceKernel2 << <IMG_HEIGHT, IMG_WIDTH >> > (color, 0);
            }
            cudaDeviceSynchronize();
            cudaMemcpy(hostcolor, color, IMG_WIDTH * IMG_HEIGHT * sizeof(PWVector3f), cudaMemcpyDeviceToHost);

            /* Release */
            cudaFree(color);

            return err;
        }
    }
}
