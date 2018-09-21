#pragma once
#include <stdafx.h>

#include <Core/Graphics.hpp>

#pragma warning(push)
#pragma warning(disable : 4005)
#include <D3D11.h>
#include <D3DX10math.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx10.lib")
#pragma warning(pop)

#include <DX11/ModelDX11.hpp>
#include <DX11/Font.hpp>

namespace Quin::System::DX11
{
    struct Camera
    {
        Camera() :m_x(0), m_y(0), m_z(0), m_rotx(0), m_roty(0), m_rotz(0) {}
        void Get(D3DXMATRIX& m)
        {
            if (m_dirty)
            {
                D3DXVECTOR3 pos(m_x, m_y, m_z), up(0.0f, 1.0f, 0.0f), lookAt(0.0f, 0.0f, 1.0f);
                D3DXMATRIX rotMatrix{};
                D3DXMatrixRotationYawPitchRoll(&rotMatrix, m_roty, m_rotx, m_rotz);
                D3DXVec3TransformCoord(&lookAt, &lookAt, &rotMatrix);
                D3DXVec3TransformCoord(&up, &up, &rotMatrix);
                lookAt = pos + lookAt;
                D3DXMatrixLookAtLH(&m_matrix, &pos, &lookAt, &up);
                m_dirty = false;
            }
            m = m_matrix;
        }
        D3DXVECTOR4 Pos()
        {
            return D3DXVECTOR4(m_x, m_y, m_z, 1.0f);
        }
        BOOL m_dirty = true;
        union
        {
            D3DXVECTOR3 pos;
            struct
            {
                FLOAT m_x, m_y, m_z;
            };
        };
        union
        {
            D3DXVECTOR3 rot;
            struct
            {
                FLOAT m_rotx, m_roty, m_rotz;
            };
        };
        D3DXMATRIX m_matrix;
    };

    struct Light
    {
        D3DXVECTOR3 m_dir;
    };

    class GraphicsDX11 : public Core::Graphics
    {
    public:
        GraphicsDX11() = default;
        virtual ~GraphicsDX11() = default;

        /* Override */
        virtual void Initialize(HWND hwnd, UINT w, UINT h);
        virtual void Shutdown();
        virtual BOOL OnUpdate();
        virtual LRESULT CALLBACK MessageHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    private:
        /* D3D Basic */
        void InitializeDevice(HWND hwnd);
        void ShutdownDevice();

        /* Output Merger */
        void InitializeOM();
        void ShutdownOM();

        /* Output Merger */
        void InitializeRasterizer();
        void ShutdownRasterizer();

        void BeginScene()
        {
            float color[] = { 0.25f, 0.1f, 0.15f, 0.0f };
            m_context->ClearRenderTargetView(m_RTV, color);
            m_context->ClearDepthStencilView(m_DSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
        }
        void EndScene()
        {
            m_swapChain->Present(1, 0);
        }
        void OnRender();
        void OnGUI();

        /* D3D Basic */
        UINT m_width = 640;
        UINT m_height = 480;
        IDXGISwapChain *m_swapChain = nullptr;
        ID3D11Device *m_device = nullptr;
        ID3D11DeviceContext *m_context = nullptr;

        /* OM Dynamic */
        ID3D11RenderTargetView *m_RTV = nullptr;
        ID3D11DepthStencilView *m_DSV = nullptr;
        ID3D11DepthStencilState *m_DSSWithZ = nullptr;
        ID3D11DepthStencilState *m_DSSWithoutZ = nullptr;
        ID3D11BlendState *m_BSWithBlend = nullptr;
        ID3D11BlendState *m_BSWithoutBlend = nullptr;

        /* RS Dynamic */
        // No Components Will Change For Now

        /* Projection Matrix */
        D3DXMATRIX m_MatrixProj{};
        D3DXMATRIX m_MatrixOrtho{};

        Camera m_camera;
        ModelDX11 *m_model = nullptr;
        Font *m_gui = nullptr;
        Light m_light;
    };
}