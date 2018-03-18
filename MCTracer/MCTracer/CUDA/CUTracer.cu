#include "CUTracer.h"
#include <cublas.h>
#include <thrust/device_vector.h>
#include <vector>

#include "Framework/Geometry.h"

namespace PW
{
    namespace Tracer
    {
        __device__ PWVector3f* vertexBuffer;
        __device__ PWuint nVertexBuffer;
        __device__ PWVector3f* normalBuffer;
        __device__ PWuint nNormalBuffer;
        __device__ Geometry::Triangle* triangleBuffer;
        __device__ PWuint nTriangleBuffer;
        __device__ Geometry::Geometry* geometryBuffer;
        __device__ PWuint nGeometryBuffer;

        __device__ int* a;
        __device__ int* b;

        __global__ void addKernel(PWVector3f *c)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            c[i] = geometryBuffer[i].material.Kd;
        }

        __global__ void kernel(int *c)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            c[i] = a[i] + b[i];
        }

        cudaError_t RenderScene1(PW::FileReader::ObjModel *model)
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
            cudaMemcpy(deviceVertexBufferAddr, &hostNormalBuffer[0], sizeof(PWVector3f) * deviceNormalBufferNum, cudaMemcpyHostToDevice);

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

            PWVector3f *color = nullptr;
            cudaStatus = cudaMalloc((void**)&color, 4 * sizeof(PWVector3f));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                cudaFree(color);
                return cudaStatus;
            }
            // Launch a kernel on the GPU with one thread for each element.
            addKernel << <1, 4 >> > (color);
            PWVector4f *hostcolor = new PWVector4f(800 * 600 * 1024); // Width*Height*Samples
            cudaMemcpy(hostcolor, color, 4 * sizeof(PWVector3f), cudaMemcpyDeviceToHost);

            void* devAddr = nullptr;
            cudaGetSymbolAddress(&devAddr, a);
            std::vector<int> hosta(4);
            for (int i = 0; i < 4; i++) hosta[i] = i;
            cudaMalloc((void**)&devAddr, sizeof(int) * 4);
            cudaMemcpy(a, &hosta[0], sizeof(int) * 4, cudaMemcpyHostToDevice);
            std::vector<int> hostb(4);
            for (int i = 0; i < 4; i++) hostb[i] = 10 * i;
            cudaMalloc((void**)&b, sizeof(int) * 4);
            cudaMemcpy(b, &hostb[0], sizeof(int) * 4, cudaMemcpyHostToDevice);
            int *d_c = nullptr;
            cudaMalloc((void**)d_c, sizeof(int) * 4);
            kernel << <1, 4 >> > (d_c);
            int hostc[4];

            cudaDeviceSynchronize();
            cudaMemcpy(hostc, d_c, sizeof(int) * 4, cudaMemcpyDeviceToHost);
            //
            //    // Check for any errors launching the kernel
            //    cudaStatus = cudaGetLastError();
            //    if (cudaStatus != cudaSuccess) {
            //        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            //        goto Error;
            //    }
            //
            //    // cudaDeviceSynchronize waits for the kernel to finish, and returns
            //    // any errors encountered during the launch.
            //    cudaStatus = cudaDeviceSynchronize();
            //    if (cudaStatus != cudaSuccess) {
            //        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            //        goto Error;
            //    }
            //
            //    // Copy output vector from GPU buffer to host memory.
            //    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
            //    if (cudaStatus != cudaSuccess) {
            //        fprintf(stderr, "cudaMemcpy failed!");
            //        goto Error;
            //    }
            //
            //Error:
            //    cudaFree(dev_c);
            //    cudaFree(dev_a);
            //    cudaFree(dev_b);

            return cudaStatus;
        }
    }
}
