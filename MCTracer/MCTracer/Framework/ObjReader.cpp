#include "ObjReader.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

PWbool PW::FileReader::ObjModel::readObj(const std::string &path)
{
    clear();
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::ios_base::failure("Can't open file");
    }
    m_path = path;
    std::map<std::string, ObjGroup>::iterator it_group = findAndAddGroup("default");
    PWint materialIndex = 0;

    std::string token;
    std::string line;
    std::stringstream lineBuffer;
    while (std::getline(file, line))
    {
        lineBuffer.str("");
        lineBuffer.clear();
        lineBuffer.sync();
        while (line.length() > 0 && line[line.length() - 1] == '\\')
        {
            line.pop_back();
            lineBuffer << line;
            std::getline(file, line);
        }
        lineBuffer << line;

        /* Each line */
        if (lineBuffer >> token)
        {
            /* Comment */
            if (token[0] == '#')
            {
                /* Ignore */
            }
            /* Material lib */
            else if (token == "mtllib")
            {
                lineBuffer >> token;
                readMtl(token);
            }
            /* Group */
            else if (token == "g")
            {
                lineBuffer >> token;
                it_group = findAndAddGroup(token);
            }
            /* Use material */
            else if (token == "usemtl")
            {
                lineBuffer >> token;
                materialIndex = findMaterial(token);
                it_group->second.materialIndex = materialIndex;
            }
            /* Face */
            else if (token == "f")
            {
                PWint idx = 0;
                std::stringstream tokenBuf;
                PWint vIdx, tIdx, nIdx;
                char dummyChar;
                m_triangles.emplace_back();
                it_group->second.m_triangleIndices.push_back(m_triangles.size() - 1);

                while (lineBuffer >> token)
                {
                    tokenBuf.str("");
                    tokenBuf.clear();
                    tokenBuf.sync();
                    tokenBuf << token;

                    if (!parseFaceVertex(tokenBuf, vIdx, tIdx, nIdx))
                    {
                        throw "Invalid OBJ file!";
                    }

                    if (idx < 3)
                    {
                        m_triangles.back().m_vertexIndex[idx] = vIdx;
                        m_triangles.back().m_textureIndex[idx] = tIdx;
                        m_triangles.back().m_normalIndex[idx] = nIdx;
                    }
                    else
                    {
                        m_triangles.emplace_back();
                        it_group->second.m_triangleIndices.push_back(m_triangles.size() - 1);
                        m_triangles.back().m_vertexIndex[0] = m_triangles[m_triangles.size() - 2].m_vertexIndex[0];
                        m_triangles.back().m_vertexIndex[1] = m_triangles[m_triangles.size() - 2].m_vertexIndex[2];
                        m_triangles.back().m_vertexIndex[2] = vIdx;
                        m_triangles.back().m_textureIndex[0] = m_triangles[m_triangles.size() - 2].m_textureIndex[0];
                        m_triangles.back().m_textureIndex[1] = m_triangles[m_triangles.size() - 2].m_textureIndex[2];
                        m_triangles.back().m_textureIndex[2] = tIdx;
                        m_triangles.back().m_normalIndex[0] = m_triangles[m_triangles.size() - 2].m_normalIndex[0];
                        m_triangles.back().m_normalIndex[1] = m_triangles[m_triangles.size() - 2].m_normalIndex[2];
                        m_triangles.back().m_normalIndex[2] = nIdx;
                    }
                    ++idx;
                }
            }
            /* Vertex */
            else if (token == "v")
            {
                ///TODO
                PWdouble x, y, z;
                lineBuffer >> x >> y >> z;
                m_vertices.emplace_back(x, y, z);
            }
        }
    }
    file.close();

    /* normalize */
    PW::Math::Vector3d meanV;
    PWdouble minX = m_vertices[1].getX();
    PWdouble minY = m_vertices[1].getY();
    PWdouble minZ = m_vertices[1].getZ();
    PWdouble maxX = m_vertices[1].getX();
    PWdouble maxY = m_vertices[1].getY();
    PWdouble maxZ = m_vertices[1].getZ();
    meanV.setX(0);
    meanV.setY(0);
    meanV.setZ(0);
    for (int i = 1; i < m_vertices.size(); ++i)
    {
        minX = std::min(minX, m_vertices[i].getX());
        minY = std::min(minY, m_vertices[i].getY());
        minZ = std::min(minZ, m_vertices[i].getZ());
        maxX = std::max(maxX, m_vertices[i].getX());
        maxY = std::max(maxY, m_vertices[i].getY());
        maxZ = std::max(maxZ, m_vertices[i].getZ());
        meanV += m_vertices[i];
    }
    meanV /= m_vertices.size() - 1;
    PWdouble width = std::max(maxX - minX, std::max(maxY - minY, maxZ - minZ));
    for (int i = 1; i < m_vertices.size(); ++i)
    {
        m_vertices[i] = (m_vertices[i] - meanV) / width * 25;
    }
    return true;
}

PWbool PW::FileReader::ObjModel::readMtl(const std::string & path)
{
    return true;
}
