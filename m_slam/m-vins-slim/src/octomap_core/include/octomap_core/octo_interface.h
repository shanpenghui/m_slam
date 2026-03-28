#ifndef MVINS_OCTOMAP_CORE_OCCUPANCY_INTERFACE_H_
#define MVINS_OCTOMAP_CORE_OCCUPANCY_INTERFACE_H_

#include <memory>

#include <octomap/ColorOcTree.h>

namespace octomap {

class OctomapInterface {
public:
    explicit OctomapInterface(const double resolution);
    ~OctomapInterface() = default;
    void SetOcTree(const std::shared_ptr<octomap::ColorOcTree>& octree_ptr);
    std::shared_ptr<octomap::ColorOcTree> GetOcTree();
private:
    const double resolution_;
    std::shared_ptr<octomap::ColorOcTree> octree_;

};

}

#endif
