import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rigid_estimate as rie

def compute_rmse(input_slam_file, input_gt_file):

    trajectory_slam = pd.read_csv('tmp/sync_slam.csv', header=None, names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    trajectory_gt = pd.read_csv('tmp/sync_gt.csv', header=None, names=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    slam_x = trajectory_slam['x']
    slam_y = trajectory_slam['y']
    slam_z = trajectory_slam['z']
    slam_qx = trajectory_slam['qx']
    slam_qy = trajectory_slam['qy']
    slam_qz = trajectory_slam['qz']
    slam_qw = trajectory_slam['qw']

    gt_x = trajectory_gt['x']
    gt_y = trajectory_gt['y']
    gt_z = trajectory_gt['z']
    gt_qx = trajectory_gt['qx']
    gt_qy = trajectory_gt['qy']
    gt_qz = trajectory_gt['qz']
    gt_qw = trajectory_gt['qw']

    np_slam = np.array(trajectory_slam)
    np_gt = np.array(trajectory_gt)

    ts = range(len(slam_x))

    slam_R = Rotation.from_quat(np.column_stack((slam_qx, slam_qy, slam_qz, slam_qw))).as_matrix()
    gt_R = Rotation.from_quat(np.column_stack((gt_qx, gt_qy, gt_qz, gt_qw))).as_matrix()

    aligned_gt_R, aligned_gt_t = rie.rigid_transform_3D(np_gt[:, :3], np_slam[:, :3])
    print(f"R: {aligned_gt_R}")
    print(f"t: {aligned_gt_t}")
    aligned_gt_trajectory = np_gt[:, :3]
    for i in range(len(aligned_gt_trajectory)):
        aligned_gt_trajectory[i] = aligned_gt_R @ np_gt[i, :3] + aligned_gt_t

    rmse = np.sqrt(np.mean((aligned_gt_trajectory - np_slam[:, :3]) ** 2, axis=0))

    print(f"RMSE: {rmse}")

    plt.plot(aligned_gt_trajectory[:,0], aligned_gt_trajectory[:,1], color='blue', label='Ground Truth')
    plt.plot(np_slam[:,0], np_slam[:,1], color='red', label='SLAM')
    for i in range(len(aligned_gt_trajectory)):
        if (i % 100 == 0):
            plt.plot([aligned_gt_trajectory[i,0], np_slam[i,0]],
                    [[aligned_gt_trajectory[i,1]], [np_slam[i,1]]], color ='green')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RMSE: ' + str(rmse))

    plt.show()