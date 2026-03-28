#include "octomap_core/octo_interface.h"

#include <glog/logging.h>

#include "data_common/constants.h"
#include "file_common/file_system_tools.h"

namespace octomap {

OctomapInterface::OctomapInterface(const double resolution)
    : resolution_(resolution), octree_(nullptr) {
}

void OctomapInterface::SetOcTree(const std::shared_ptr<octomap::ColorOcTree>& octree_ptr) {
    CHECK_NOTNULL(octree_ptr);
    octree_ = octree_ptr;
}

std::shared_ptr<octomap::ColorOcTree> OctomapInterface::GetOcTree() {
    if (octree_ != nullptr) {
        return octree_;
    }
    return nullptr;
}
}
