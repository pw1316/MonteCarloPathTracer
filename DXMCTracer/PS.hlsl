struct PSin
{
    float4 color: COLOR;
};

float4 PSMain(PSin input) : SV_TARGET
{
    return input.color;
}