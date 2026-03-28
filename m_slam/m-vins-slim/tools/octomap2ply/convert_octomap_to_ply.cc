#include <iostream>
#include <fstream>
#include <pcl/io/ply_io.h>
#include <octomap/OcTree.h>
#include <octomap/ColorOcTree.h>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_octomap_file> <output_ply_file>" << std::endl;
        return 1;
    }

    // 读取OctoMap文件
    octomap::AbstractOcTree* tree = octomap::AbstractOcTree::read(argv[1]);
    if (tree == nullptr)
    {
        std::cerr << "Error: Failed to read OctoMap file." << std::endl;
        return 1;
    }

    // 检查OctoMap版本是否为彩色版本
    octomap::ColorOcTree* color_tree = dynamic_cast<octomap::ColorOcTree*>(tree);
    bool is_color_tree = (color_tree != nullptr);
    if (!is_color_tree)
    {
        std::cerr << "Error: Input OctoMap is not a color OctoMap." << std::endl;
        return 1;
    }

    // 提取点云
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    for (octomap::ColorOcTree::leaf_iterator it = color_tree->begin_leafs(); it != color_tree->end_leafs(); ++it)
    {
        if (color_tree->isNodeOccupied(*it))
        {
            octomap::point3d point = it.getCoordinate();
            octomap::ColorOcTreeNode* node = color_tree->search(it.getKey());
            pcl::PointXYZRGB pcl_point(node->getColor().r, node->getColor().g, node->getColor().b);
            pcl_point.x = point.x();
            pcl_point.y = point.y();
            pcl_point.z = point.z();
            cloud.push_back(pcl_point);
        }
    }

    // 将点云保存为PLY文件
    pcl::PLYWriter writer;
    writer.write(argv[2], cloud, true, false);

    return 0;
}

