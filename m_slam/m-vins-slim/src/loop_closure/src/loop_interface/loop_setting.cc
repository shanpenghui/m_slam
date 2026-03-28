#include "loop_interface/loop_settings.h"

#include <glog/logging.h>

#include "file_common/file_system_tools.h"

namespace loop_closure {

LoopSettings::LoopSettings(const common::SlamConfigPtr& config)
    : asserts_file_path(config->assets_path),
      num_closest_words_for_nn_search(config->num_words_for_nn_search),
      min_image_time_seconds(config->min_image_time_seconds),
      min_verify_matches_num(config->min_verify_matches_num),
      fraction_best_scores(config->fraction_best_scores),
      min_inlier_count(config->min_inlier_count),
      cached_num_threads(config->hardware_threads),
      num_nearest_neighbors(config->num_neighbors),
      loop_closure_sigma_pixel(config->outlier_rejection_scale * config->visual_sigma_pixel),
      local_search_radius(config->local_search_radius),
      local_search_angles(config->local_search_angles),
      pnp_num_ransac_iters(config->pnp_num_ransac_iters),
      max_knn_radius(config->max_knn_radius) {
    CHECK_GT(num_closest_words_for_nn_search, 0);
    CHECK_GE(min_image_time_seconds, 0.0);
    CHECK_GE(min_verify_matches_num, 0);
    CHECK_GT(fraction_best_scores, 0.f);
    CHECK_LT(fraction_best_scores, 1.f);
    CHECK_GE(num_nearest_neighbors, -1);

    SetKeyframeScoringFunctionType(config->scoring_function);
    projected_quantizer_filename = common::ConcatenateFilePathFrom(
                asserts_file_path,
                "inverted_multi_index_quantizer_freak.dat");
}

void LoopSettings::SetKeyframeScoringFunctionType(
    const std::string& scoring_function_string) {
  scoring_function_type_string = scoring_function_string;
  if (scoring_function_string == kAccumulationString) {
    keyframe_scoring_function_type = KeyframeScoringFunctionType::kAccumulation;
  } else if (scoring_function_string == kProbabilisticString) {
    keyframe_scoring_function_type =
        KeyframeScoringFunctionType::kProbabilistic;
  } else {
    LOG(FATAL) << "Unknown scoring function type: " << scoring_function_string;
  }
}

}
