cbuffer consts0 : register(b0)
{
    matrix MatrixWorld;
    matrix MatrixView;
    matrix MatrixProj;
};

cbuffer consts1 : register(b1)
{
    float4 CameraPos;
    float3 LightDir;
};

struct VertexIn
{
    float4 pos : POSITION;
    float2 uv : TEXCOORD0;
    float3 normal : NORMAL;
};

struct VertexOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float3 normal : NORMAL;
    float3 view : TEXCOORD1;
};

VertexOut VS(VertexIn vin)
{
    VertexOut vout;
    vin.pos.w = 1;
    float4 pos;
    pos = mul(vin.pos, MatrixWorld);
    vout.view = normalize((CameraPos - pos).xyz);
    pos = mul(pos, MatrixView);
    vout.pos = mul(pos, MatrixProj);
    vout.uv = vin.uv;
    vout.normal = normalize(mul(vin.normal, (float3x3) MatrixWorld));
    return vout;
}