#ifndef MVINS_DELAUNAY_MESHER_H_
#define MVINS_DELAUNAY_MESHER_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace vins_core {

struct Vertex {
    // Constructors.
    Vertex() : x(0.), y(0.), index(-1) {}

    Vertex(const double _x, const double _y)
        : x(_x), y(_y), index(-1) {}

    Vertex(const double _x, const double _y, const int _index)
        : x(_x), y(_y), index(_index) {}


    // Compute the squared distance between the two vertices.
    double Dist2(const Vertex& v) const {
        const double dx = x - v.x;
        const double dy = y - v.y;
        return dx * dx + dy * dy;
    }

    // Compute the distance between the two vertices.
    double Dist(const Vertex& v) const {
        return std::hypot(x - v.x, y - v.y);
    }

    // Compute the norm of the vertex from origin.
    double Norm2() const {
        return x * x + y * y;
    }

    // Operator overloads.
    bool operator ==(const Vertex& v) const {
        return (x == v.x) && (y == v.y) && (index == v.index);
    }

    friend std::ostream& operator <<(std::ostream& str, const Vertex& v) {
        return str << "Point x: " << v.x << " y: " << v.y;
    }

    // Member variables.
    double x;
    double y;
    int index;
};

struct Edge {
    Edge(const Vertex& _v1, const Vertex& _v2)
        : v1(&_v1), v2(&_v2), is_bad(false) {}

    bool operator ==(const Edge& e) const {
        return (*v1 == *e.v1 && *v2 == *e.v2) ||
               (*v1 == *e.v2 && *v2 == *e.v1);
    }

    friend std::ostream& operator <<(std::ostream& str, const Edge& e) {
        return str << "Edge " << *e.v1 << ", " << *e.v2;
    }

    const Vertex* v1;
    const Vertex* v2;
    bool is_bad;
};

struct Triangle {
    Triangle(const Vertex& _v1,
             const Vertex& _v2,
             const Vertex& _v3)
             : v1(&_v1), v2(&_v2), v3(&_v3), is_bad(false) {}

    // Check whether v is one of the vertices of the triangle.
    bool ContainVertex(const Vertex& v) const;

    // Check whether v is inside the circumscribed circle.
    bool CircumCircleContains(const Vertex& v) const;

    bool operator ==(const Triangle& t) const {
        return (*v1 == *t.v1 || *v1 == *t.v2 || *v1 == *t.v3) &&
               (*v2 == *t.v1 || *v2 == *t.v2 || *v2 == *t.v3) &&
               (*v3 == *t.v1 || *v3 == *t.v2 || *v3 == *t.v3);
    }

    friend std::ostream& operator <<(std::ostream& str, const Triangle& t) {
        return str << "Triangle:" << "\n\t" <<
                   *t.v1 << "\n\t" <<
                   *t.v2 << "\n\t" <<
                   *t.v3 << '\n';
    }

    const Vertex* v1;
    const Vertex* v2;
    const Vertex* v3;
    bool is_bad;
};

class DelaunayMeshGenerator {
 public:
    DelaunayMeshGenerator() = default;

    // Generate mesh using the input list of vertices.
    const std::vector<Triangle>& GenerateMesh(
            const std::vector<Vertex>& _vertices);

    // Get meshing results.
    const std::vector<Triangle>& GetTriangles() const;
    const std::vector<Edge>& GetEdges() const;
    const std::vector<Vertex>& GetVertices() const;

 private:
    std::vector<Triangle> triangles_;
    std::vector<Edge> edges_;
    std::vector<Vertex> vertices_;
};

}  // namespace vins_core

#endif  // MVINS_DELAUNAY_MESHER_H_
