#define CS_PI 3.14159265358979323846f
#define CS_FLT_EPSILON 1.192092896e-07f

cbuffer consts0 : register(b0)
{
    matrix viewMatrix;
    matrix projMatrix;
    uint seed;
    uint prevCount;
    uint width;
    uint height;
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

struct CSKDTree
{
    uint left, right;
    uint parent;
    float3 aabbMin;
    float3 aabbMax;
    uint splitAxis;
    float splitValue;
    uint triIds[64];
    uint numTri;
};

struct HitInfo
{
    uint triID;
    float beta;
    float gamma;
    float3 hitPoint;
};

StructuredBuffer<float3> csvertex : register(t0);
StructuredBuffer<float3> csnormal : register(t1);
StructuredBuffer<CSTriangle> cstriangle : register(t2);
StructuredBuffer<CSKDTree> cskdtree : register(t3);
StructuredBuffer<CSMaterial> csmaterial : register(t4);
Texture2D<float4> csscreen_r : register(t5);

RWTexture2D<float4> csscreen_w : register(u0);
RWTexture2D<float4> csrtv : register(u1);

uint rand_seed(uint v0, uint v1, uint backoff = 16)
{
    uint sum = 0;

    for (uint n = 0; n < backoff; n++)
    {
        sum += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

float rand_gen(inout uint sd)
{
    sd = 16807 * (sd % 127773) - 2836 * (sd / 127773);
    if (sd > 0x7FFFFFFF)
    {
        sd += 0x7FFFFFFF;
    }
    return 1.0f * sd / 0x7FFFFFFF;
}

bool intersect_aabb(float3 from, float3 dir, uint nodeId, float curt)
{
    float t_min, t_max, t_xmin, t_xmax, t_ymin, t_ymax, t_zmin, t_zmax;
    float x_a = 1.0 / dir.x, y_a = 1.0 / dir.y, z_a = 1.0 / dir.z;
    float x_e = from.x, y_e = from.y, z_e = from.z;
    if (dir.x == 0 && (from.x < cskdtree[nodeId].aabbMin.x || from.x > cskdtree[nodeId].aabbMax.x))
    {
        return false;
    }
    if (dir.y == 0 && (from.y < cskdtree[nodeId].aabbMin.y || from.y > cskdtree[nodeId].aabbMax.y))
    {
        return false;
    }
    if (dir.z == 0 && (from.z < cskdtree[nodeId].aabbMin.z || from.z > cskdtree[nodeId].aabbMax.z))
    {
        return false;
    }

	// calculate t interval in x-axis
    if (x_a >= 0)
    {
        t_xmin = (cskdtree[nodeId].aabbMin.x - x_e) * x_a;
        t_xmax = (cskdtree[nodeId].aabbMax.x - x_e) * x_a;
    }
    else
    {
        t_xmin = (cskdtree[nodeId].aabbMax.x - x_e) * x_a;
        t_xmax = (cskdtree[nodeId].aabbMin.x - x_e) * x_a;
    }

	// calculate t interval in y-axis
    if (y_a >= 0)
    {
        t_ymin = (cskdtree[nodeId].aabbMin.y - y_e) * y_a;
        t_ymax = (cskdtree[nodeId].aabbMax.y - y_e) * y_a;
    }
    else
    {
        t_ymin = (cskdtree[nodeId].aabbMax.y - y_e) * y_a;
        t_ymax = (cskdtree[nodeId].aabbMin.y - y_e) * y_a;
    }

	// calculate t interval in z-axis
    if (z_a >= 0)
    {
        t_zmin = (cskdtree[nodeId].aabbMin.z - z_e) * z_a;
        t_zmax = (cskdtree[nodeId].aabbMax.z - z_e) * z_a;
    }
    else
    {
        t_zmin = (cskdtree[nodeId].aabbMax.z - z_e) * z_a;
        t_zmax = (cskdtree[nodeId].aabbMin.z - z_e) * z_a;
    }

	// find if there an intersection among three t intervals
    t_min = max(t_xmin, max(t_ymin, t_zmin));
    t_max = min(t_xmax, min(t_ymax, t_zmax));
    return (t_min <= t_max && t_min <= curt);
}

HitInfo intersect(float3 from, float3 dir)
{
    HitInfo hitInfo;
    hitInfo.triID = -1;

    float tmin = 10000;
    
    uint localstack[64];
    uint stacktop = 0;
    localstack[stacktop++] = 0;
	[allow_uav_condition]
    while (stacktop != 0)
    {
        uint node = localstack[--stacktop];
        if (intersect_aabb(from, dir, node, tmin))
        {
            if (cskdtree[node].splitAxis == 0)
            {
                for (uint i = 0; i < cskdtree[node].numTri; i++)
                {
                    float3 a = csvertex[cstriangle[cskdtree[node].triIds[i]].v0];
                    float3 b = csvertex[cstriangle[cskdtree[node].triIds[i]].v1];
                    float3 c = csvertex[cstriangle[cskdtree[node].triIds[i]].v2];
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
                        hitInfo.triID = cskdtree[node].triIds[i];
                        hitInfo.beta = beta;
                        hitInfo.gamma = gamma;
                        hitInfo.hitPoint.x = from.x + t * dir.x;
                        hitInfo.hitPoint.y = from.y + t * dir.y;
                        hitInfo.hitPoint.z = from.z + t * dir.z;
                    }
                }
            }
            else
            {
                localstack[stacktop++] = cskdtree[node].left;
                localstack[stacktop++] = cskdtree[node].right;
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
    uint bounce = 0;
    hit = intersect(from, dir);
	[allow_uav_condition]
    while (hit.triID != -1)
    {
        if (bounce >= depth * 3)
        {
            return float3(0, 0, 0);
        }
        else if (bounce >= depth)
        {
            float illum = max(max(color.x, color.y), color.z);
            if (illum > rand_gen(sd))
            {
                color /= illum;
            }
            else
            {
                return float3(0, 0, 0);
            }
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
            //color *= csmaterial[matId].Kd;
        }
		/* Specular */
        else if (csmaterial[matId].Ns > 1)
        {
            dir = samplePhong(sd, normal, dir, csmaterial[matId].Ns);
            color *= csmaterial[matId].Ks;
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
        }
        from = hit.hitPoint + dir * 0.01f;
        hit = intersect(from, dir);
        ++bounce;
    }
    return float3(0, 0, 0);
}

[numthreads(32, 32, 1)]
void main(uint3 gId : SV_GroupId, uint3 tId : SV_GroupThreadId)
{
    uint2 uv;
    uv.x = (gId.x * 32 + tId.x);
    uv.y = (gId.y * 32 + tId.y);
	
    uint sss = rand_seed(uv.y * width + uv.x, seed);
    rand_gen(sss);
    rand_gen(sss);
	
    /* Bias */
    float biasx = uv.x + (rand_gen(sss) - 0.5);
    float biasy = uv.y + (rand_gen(sss) - 0.5);

    /* Reproject */
    float3 ray_from = float3(0, 0, 0);
    float3 ray_dir = float3(
		(2.0 * biasx / width - 1) / projMatrix._11,
        (1 - 2.0 * biasy / height) / projMatrix._22,
        -1
    );
    /* View to World */
    ray_from = mul(float4(ray_dir, 1), viewMatrix).xyz;
    ray_dir = mul(float4(ray_dir, 0), viewMatrix).xyz;
    ray_dir = normalize(ray_dir);
	
    float3 color = sampleMC(sss, ray_from, ray_dir, 5);
    float4 oldcolor = csscreen_r[uv];
    float4 newcolor = pow((pow(oldcolor, 2.2) * prevCount + float4(color, 1)) / (prevCount + 1), 1 / 2.2);
    csscreen_w[uv] = newcolor;
    csrtv[uv] = newcolor;
}
