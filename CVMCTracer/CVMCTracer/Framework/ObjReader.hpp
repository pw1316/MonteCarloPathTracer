#pragma once
#include <exception>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include "Math.hpp"

namespace PW {
    namespace FileReader
    {
        struct ObjTriangle
        {
            PWint m_vertexIndex[3];
            PWint m_textureIndex[3];
            PWint m_normalIndex[3];
            PWint materialIndex;
        };

        struct ObjMaterial
        {
            ObjMaterial(const std::string &n) :name(n), Ka(0.0, 0.0, 0.0), Kd(0.0, 0.0, 0.0), Ks(0, 0, 0), Ns(1), Tr(0), Ni(1) {};
            std::string name;
            Math::Vector3d Ka;
            Math::Vector3d Kd;
            Math::Vector3d Ks;
            PWdouble Ns;
            PWdouble Tr;
            PWdouble Ni;
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
                m_textures.clear();
                m_textures.emplace_back(0, 0);
                m_normals.clear();
                m_normals.emplace_back(0, 0, 0);
                m_triangles.clear();
                m_triangles.emplace_back();
                m_materials.clear();
                m_materials.emplace_back("");
                m_groups.clear();
            }
            PWbool readObj(const std::string &path);

            std::string m_path;
            std::vector<Math::Vector3d> m_vertices;
            std::vector<Math::Vector2d> m_textures;
            std::vector<Math::Vector3d> m_normals;
            std::vector<ObjTriangle> m_triangles;
            std::vector<ObjMaterial> m_materials;
            std::map<std::string, ObjGroup> m_groups;
        private:
            PWbool readMtl(const std::string &path);

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

            PWuint findMaterial(const std::string &name)
            {
                for (size_t i = 1; i < m_materials.size(); i++)
                {
                    if (m_materials[i].name == name)
                    {
                        return (PWuint)i;
                    }
                }
                return 0;
            }

            PWbool parseFaceVertex(std::stringstream &buffer, PWint &vertexIdx, PWint &textureIdx, PWint &normalIdx)
            {
                char dummy;
                /* v */
                buffer >> vertexIdx;
                if (buffer.fail())
                {
                    return false;
                }
                buffer >> dummy;
                /* no t and no n */
                if (buffer.fail())
                {
                    textureIdx = 0;
                    normalIdx = 0;
                    return true;
                }
                buffer >> textureIdx;
                /* no t */
                if (buffer.fail())
                {
                    textureIdx = 0;
                    buffer.clear();
                    buffer >> dummy;
                    buffer >> normalIdx;
                    /* no n */
                    if (buffer.fail())
                    {
                        return false;
                    }
                    /* with n */
                    return true;
                }
                /* with t */
                buffer >> dummy;
                /* no n */
                if (buffer.fail())
                {
                    normalIdx = 0;
                    return true;
                }
                /* with n */
                buffer >> normalIdx;
                if (buffer.fail())
                {
                    return false;
                }
                return true;
            }
        };
    }
}