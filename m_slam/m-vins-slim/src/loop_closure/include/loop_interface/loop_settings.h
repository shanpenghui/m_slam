#ifndef LOOP_CLOSURE_LOOP_SETTINGS_H_
#define LOOP_CLOSURE_LOOP_SETTINGS_H_

#include <string>

#include "cfg_common/slam_config.h"

namespace loop_closure {

static std::string kAccumulationString = "accumulation";
static std::string kProbabilisticString = "probabilistic";

struct LoopSettings {
  LoopSettings(const common::SlamConfigPtr& config);

  enum class KeyframeScoringFunctionType {
    kAccumulation,
    kProbabilistic,
  };

  void SetKeyframeScoringFunctionType(
      const std::string& scoring_function_string);

  KeyframeScoringFunctionType keyframe_scoring_function_type;
  std::string scoring_function_type_string;

  std::string asserts_file_path;
  std::string projected_quantizer_filename;
  int num_closest_words_for_nn_search;
  double min_image_time_seconds;
  int min_verify_matches_num;
  float fraction_best_scores;
  int min_inlier_count;
  size_t cached_num_threads;
  int num_nearest_neighbors;
  double loop_closure_sigma_pixel;
  double local_search_radius;
  double local_search_angles;
  int pnp_num_ransac_iters;
  float max_knn_radius;

};

}

#endif
