cbuffer cbPerObject
{
    matrix mWVP;
};

struct VSin
{
    float4 pos : POSITION;
    float4 color : COLOR;
};

struct VSout
{
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

VSout VSMain(VSin input)
{
    VSout output;
    output.pos = mul(input.pos, mWVP);
    output.color = input.color;
    return output;
}