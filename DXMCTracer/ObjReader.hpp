#pragma once
#include <exception>
#include <map>
#include <string>
#include <vector>
#include "Math.hpp"

namespace FileReader
{
    struct ObjTriangle
    {
        PWint m_vertexIndex[3];
    };

    struct ObjGroup
    {
        std::vector<PWint> m_triangleIndices;
    };

    class ObjModel
    {
    public:
        void clear()
        {
            m_path.clear();
            m_vertices.clear();
            m_vertices.emplace_back(0, 0, 0);
            m_triangles.clear();
            m_triangles.emplace_back();
            m_groups.clear();
        }
        PWbool readObj(const std::string &path);

        std::string m_path;
        std::vector<Math::Vector3d> m_vertices;
        std::vector<ObjTriangle> m_triangles;
        std::map<std::string, ObjGroup> m_groups;
    private:
        std::map<std::string, ObjGroup>::iterator findAndAddGroup(const std::string &name)
        {
            std::map<std::string, ObjGroup>::iterator it = m_groups.find(name);
            if (it == m_groups.end())
            {
                m_groups[name] = ObjGroup();
                it = m_groups.find(name);
            }
            return it;
        }
    };
}