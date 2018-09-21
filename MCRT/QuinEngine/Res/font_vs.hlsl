cbuffer consts0 : register(b0)
{
    matrix MatrixProj;
};

struct VertexIn
{
    float4 pos : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

VertexOut VS(VertexIn vin)
{
    VertexOut vout;
    vout.pos = mul(vin.pos, MatrixProj);
    vout.uv = vin.uv;
    return vout;
}
