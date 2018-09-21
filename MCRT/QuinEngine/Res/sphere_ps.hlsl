Texture2D shaderTexture : register(t0);
SamplerState SampleType;

cbuffer consts0 : register(b0)
{
    float4 Ka;
    float4 Kd;
    float4 Ks;
    float Ns;
};

cbuffer consts1 : register(b1)
{
    float4 CameraPos;
    float3 LightDir;
};

struct PixelIn
{
    float4 pos: SV_POSITION;
    float2 uv: TEXCOORD0;
    float3 normal: NORMAL;
    float3 view: TEXCOORD1;
};

float4 PS(PixelIn pin) : SV_TARGET
{
    float4 color = shaderTexture.Sample(SampleType, pin.uv);
    color = color * saturate(Ka + Kd * saturate(dot(pin.normal, -LightDir)) + Ks * pow(saturate(dot(normalize(pin.view - LightDir), pin.normal)), Ns));
    return color;
}