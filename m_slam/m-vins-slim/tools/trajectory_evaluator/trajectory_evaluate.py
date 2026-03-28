import argparse
import split_log as spl
import detect_motion as dtm
import sync_pose_pair as spp
import compute_rmse as cor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--slam_log", help="the log path of SLAM")
    parser.add_argument("--gt_log", help="the trajectory file of ground truth")
    args = parser.parse_args()

    spl.split_log(args.slam_log, 'tmp/slam.csv', 'PoseEvaluate', ',')
    spl.split_log(args.gt_log, 'tmp/gt.csv', 'GroundTruthEvaluate', ',')

    slam_start_time = dtm.detect_motion('tmp/slam.csv')
    gt_start_time = dtm.detect_motion('tmp/gt.csv')

    spp.sync_pose_pair('tmp/slam.csv', 'tmp/gt.csv', slam_start_time, gt_start_time, 'tmp/sync_slam.csv', 'tmp/sync_gt.csv')

    cor.compute_rmse('tmp/sync_slam.csv', 'tmp/sync_gt.csv')
