#define CS_PI 3.14159265358979323846f
#define CS_FLT_EPSILON 1.192092896e-07f

cbuffer consts0 : register(b0)
{
    matrix viewMatrix;
    matrix projMatrix;
    uint seed;
    uint prevCount;
    uint2 padding;
};

struct CSMaterial
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float Ns;
    float Tr;
    float Ni;
};

struct CSTriangle
{
    uint v0, v1, v2;
    uint n0, n1, n2;
    uint matId;
};

struct CSGeometry
{
    uint startTri;
    uint numTries;
};

struct HitInfo
{
    uint objID;
    uint triID;
    float beta;
    float gamma;
    float3 hitPoint;
};

StructuredBuffer<float3> csvertex : register(t0);
StructuredBuffer<float3> csnormal : register(t1);
StructuredBuffer<CSTriangle> cstriangle : register(t2);
StructuredBuffer<CSGeometry> csgeometry : register(t3);
StructuredBuffer<CSMaterial> csmaterial : register(t4);

RWTexture2D<float4> csscreen : register(u0);

groupshared float sdata[1024];

float rand_gen(inout uint sd)
{
    sd = 16807 * (sd % 127773) - 2836 * (sd / 127773);
    if (sd > 0x7FFFFFFF)
    {
        sd += 0x7FFFFFFF;
    }
    return 1.0f * sd / 0x7FFFFFFF;
}

HitInfo intersect(float3 from, float3 dir)
{
    HitInfo hitInfo;
    hitInfo.objID = -1;

    float tmin = 10000;
    
    uint geoId = 0;
    uint geoNum = 0;
    uint triId = 0;
    uint triNum = 0;
    uint strides = 0;
    csgeometry.GetDimensions(geoNum, strides);
    cstriangle.GetDimensions(triNum, strides);
	
    for (geoId = 0; geoId < geoNum; geoId++)
    {
        uint offset = csgeometry[geoId].startTri;
        for (triId = 0; triId < csgeometry[geoId].numTries; triId++)
        {
            float3 a = csvertex[cstriangle[triId + offset].v0];
            float3 b = csvertex[cstriangle[triId + offset].v1];
            float3 c = csvertex[cstriangle[triId + offset].v2];
            matrix betaM = matrix(
				a.x - from.x, a.x - c.x, dir.x, 0,
                a.y - from.y, a.y - c.y, dir.y, 0,
                a.z - from.z, a.z - c.z, dir.z, 0,
				0, 0, 0, 1);
            matrix gammaM = matrix(
                a.x - b.x, a.x - from.x, dir.x, 0,
                a.y - b.y, a.y - from.y, dir.y, 0,
                a.z - b.z, a.z - from.z, dir.z, 0,
				0, 0, 0, 1);
            matrix tM = matrix(
                a.x - b.x, a.x - c.x, a.x - from.x, 0,
                a.y - b.y, a.y - c.y, a.y - from.y, 0,
                a.z - b.z, a.z - c.z, a.z - from.z, 0,
				0, 0, 0, 1);
            matrix A = matrix(
                a.x - b.x, a.x - c.x, dir.x, 0,
                a.y - b.y, a.y - c.y, dir.y, 0,
                a.z - b.z, a.z - c.z, dir.z, 0,
				0, 0, 0, 1);
            float detA = determinant(A);
            float beta = determinant(betaM) / detA;
            float gamma = determinant(gammaM) / detA;
            float t = determinant(tM) / detA;
            if (beta + gamma < 1 && beta > 0 && gamma > 0 && t > 0 && t < tmin)
            {
                tmin = t;
                hitInfo.objID = geoId;
                hitInfo.triID = triId + offset;
                hitInfo.beta = beta;
                hitInfo.gamma = gamma;
                hitInfo.hitPoint.x = from.x + t * dir.x;
                hitInfo.hitPoint.y = from.y + t * dir.y;
                hitInfo.hitPoint.z = from.z + t * dir.z;
            }
        }
    }
    return hitInfo;
}

float3 sampleFresnel(inout uint sd, float3 normal, float3 dir, float Tr, float Ni)
{
    float x = rand_gen(sd);
    float3 outdir;
    float ndoti = dot(dir, normal);
    Tr = Tr * (1 - pow(1 - abs(ndoti), 5));
    /* Refract */
    if (x < Tr)
    {
        /* In */
        if (ndoti <= 0)
        {
            float alpha = -ndoti / Ni - sqrt(1 - (1 - ndoti * ndoti) / Ni / Ni);
            outdir = normal * alpha + dir / Ni;
        }
        /* Out */
        else
        {
            float test = 1 - (1 - ndoti * ndoti) * Ni * Ni;
            /* Full reflect */
            if (test < 0)
            {
                outdir = dir - normal * dot(dir, normal) * 2;
            }
            /* With refract */
            else
            {
                float alpha = -ndoti * Ni + sqrt(test);
                outdir = normal * alpha + dir * Ni;
            }
        }
    }
    /* Reflect */
    else
    {
        outdir = dir - normal * dot(dir, normal) * 2;
    }
    return normalize(outdir);
}

float3 samplePhong(inout uint sd, float3 normal, float3 dir, float Ns)
{
    float x = rand_gen(sd);
    float y = rand_gen(sd);
    float cosT = pow(x, 1.0f / (Ns + 1));
    float sinT = sqrt(1 - cosT * cosT);
    float phi = 2 * CS_PI * y;
    float3 halfdir = float3(sinT * cos(phi), cosT, sinT * sin(phi));
    /* Inverse */
    if (abs(normal.y + 1) < CS_FLT_EPSILON)
    {
        halfdir = -halfdir;
    }
    /* Rotate */
    else if (abs(normal.y - 1) >= CS_FLT_EPSILON)
    {
        float3 dir = halfdir;
        float invlen = 1.0f / sqrt(1.0f - normal.y * normal.y);
        halfdir.x = (normal.z * dir.x + normal.x * normal.y * dir.z) * invlen + normal.x * dir.y;
        halfdir.y = normal.y * dir.y - dir.z / invlen;
        halfdir.z = (-normal.x * dir.x + normal.z * normal.y * dir.z) * invlen + normal.z * dir.y;
    }
    return dir - halfdir * dot(dir, halfdir) * 2;
}

float3 sampleHemi(inout uint sd, float3 normal)
{
    float x = rand_gen(sd);
    float y = rand_gen(sd);
    float sinT = sqrt(x);
    float cosT = sqrt(1 - x);
    float phi = 2 * CS_PI * y;
    float3 outdir = float3(sinT * cos(phi), cosT, sinT * sin(phi));
    /* Inverse */
    if (abs(normal.y + 1) < CS_FLT_EPSILON)
    {
        outdir = -outdir;
    }
    /* Rotate */
    else if (abs(normal.y - 1) >= CS_FLT_EPSILON)
    {
        float3 dir = outdir;
        float invlen = 1.0f / sqrt(1.0f - normal.y * normal.y);
        float len = 1.0f / invlen;
        outdir.x = (normal.z * dir.x + normal.x * normal.y * dir.z) * invlen + normal.x * dir.y;
        outdir.y = normal.y * dir.y - dir.z * len;
        outdir.z = (-normal.x * dir.x + normal.z * normal.y * dir.z) * invlen + normal.z * dir.y;
    }
    return outdir;
}

float3 sampleMC(inout uint sd, float3 from, float3 dir, uint depth)
{
    HitInfo hit;
    float3 color = float3(1, 1, 1);
    uint bounce;
    for (bounce = 0; bounce < depth; bounce++)
    {
        hit = intersect(from, dir);
        /* Not hit */
        if (hit.objID == -1)
        {
            return float3(0, 0, 0);
        }

        uint matId = cstriangle[hit.triID].matId;
        if (csmaterial[matId].Ka.x > 0 || csmaterial[matId].Ka.y > 0 || csmaterial[matId].Ka.z > 0)
        {
            color *= csmaterial[matId].Ka;
            return color;
        }

        float3 normal1 = csnormal[cstriangle[hit.triID].n0];
        float3 normal2 = csnormal[cstriangle[hit.triID].n1];
        float3 normal3 = csnormal[cstriangle[hit.triID].n2];
        float3 normal = normal1 * (1.0f - hit.beta - hit.gamma) + normal2 * hit.beta + normal3 * hit.gamma;
        normal = normalize(normal);
        /* Transparent */
        if (csmaterial[matId].Tr > 0)
        {
            dir = sampleFresnel(sd, normal, dir, csmaterial[matId].Tr, csmaterial[matId].Ni);
            color *= csmaterial[matId].Kd;
            from = hit.hitPoint + dir * 0.01f;
        }
		/* Specular */
        else if (csmaterial[matId].Ns > 1)
        {
            dir = samplePhong(sd, normal, dir, csmaterial[matId].Ns);
            color *= csmaterial[matId].Ks;
            from = hit.hitPoint + dir * 0.01f;
        }
        /* Diffuse */
        else
        {
            if (dot(dir, normal) > 0)
            {
                dir = -sampleHemi(sd, normal);
            }
            else
            {
                dir = sampleHemi(sd, normal);
            }
            color *= csmaterial[matId].Kd;
            from = hit.hitPoint + dir * 0.01f;
        }
    }
    hit = intersect(from, dir);
    if (hit.objID != -1)
    {
        color *= csmaterial[cstriangle[hit.triID].matId].Ka;
    }
    else
    {
        color = float3(0, 0, 0);
    }
    return color;
}

[numthreads(1, 1, 1)]
void main(uint3 gId : SV_GroupId, uint3 tId : SV_GroupThreadId)
{
    uint sss = gId.y * 640 + gId.x + seed;

	/* MC Sampling */
    float3 color = float3(0, 0, 0);
    for (int i = 0; i < 100; i++)
    {
        /* Bias */
        float biasx = gId.x + (rand_gen(sss) * 2.0f - 1.0f);
        float biasy = gId.y + (rand_gen(sss) * 2.0f - 1.0f);

        /* Reproject */
        float3 ray_from = float3(0, 0, 0);
        float3 ray_dir = float3(
			(2.0 * biasx / 640 - 1) / projMatrix._11,
            (1 - 2.0 * biasy / 480) / projMatrix._22,
            -1
        );
        /* View to World */
        ray_from = mul(float4(ray_dir, 1), viewMatrix).xyz;
        ray_dir = mul(float4(ray_dir, 0), viewMatrix).xyz;
        ray_dir = normalize(ray_dir);

        color += sampleMC(sss, ray_from, ray_dir, 7);
    }
    color /= 100;
    csscreen[gId.xy] *= prevCount;
    csscreen[gId.xy] += float4(color, 1);
    csscreen[gId.xy] /= (prevCount + 1);
}
