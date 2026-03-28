#ifndef LOOP_CLOSURE_VISUAL_LOOP_INTERFACE_H_
#define LOOP_CLOSURE_VISUAL_LOOP_INTERFACE_H_

#include "cfg_common/slam_config.h"

#include "loop_detector/loop_detector.h"

namespace loop_closure {

class VisualLoopInterface {
public:
    explicit VisualLoopInterface(const aslam::NCamera::ConstPtr& cameras,
                                 const common::SlamConfigPtr& config);
    ~VisualLoopInterface() = default;
    void LoadSummaryMap(const std::shared_ptr<loop_closure::SummaryMap>& summary_map_ptr);
    common::EigenVector3dVec GetMapClouds();
    void InsertFrameData(const common::KeyFrames& keyframes,
                         const common::VisualFrameDataPtrVec& frame_datas,
                         const common::FeaturePointPtrVec& features);
    bool Query(const common::VisualFrameDataPtrVec& frame_datas,
               const aslam::Transformation& T_GtoM,
               common::LoopResult* result_ptr,
               loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr = nullptr);
    void ProjectDescriptors(const common::DescriptorsMatUint8& raw_des,
                           common::DescriptorsMatF32* projected_des);
private:
    void LoadFrameDataFromMap();
    bool LocalizeNFrame(const common::VisualFrameDataPtrVec& frame_datas,
                         const aslam::Transformation& T_GtoM,
                         common::LoopResult* result_ptr,
                         loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr) const;
    aslam::NCamera::ConstPtr cameras_;
    std::shared_ptr<LoopSettings> settings_;
    std::shared_ptr<loop_closure::SummaryMap> summary_map_;
    std::shared_ptr<loop_closure::LoopDetector> loop_detector_;
    std::mutex mutex_;
};

}
#endif
