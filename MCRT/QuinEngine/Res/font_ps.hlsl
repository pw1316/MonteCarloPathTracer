Texture2D shaderTexture : register(t0);
SamplerState SampleType;

cbuffer consts0 : register(b0)
{
    float4 color;
};

struct PixelIn
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 PS(PixelIn pin) : SV_TARGET
{
    float4 pout;
    pout = shaderTexture.Sample(SampleType, pin.uv);
    if (pout.r == 0.0f && pout.g == 0.0f && pout.b == 0.0f)
    {
        pout.a = 0.0f;
    }
    else
    {
        pout.a = 1.0f;
        pout = pout * color;
    }
    return pout;
}
