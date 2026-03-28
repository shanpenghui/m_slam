import pandas as pd
import numpy as np

def quat2euler(quat):
    w, x, y, z = quat
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0, +1.0, t2)
    t2 = np.where(t2<-1.0, -1.0, t2)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw

def detect_motion(input_csv):

    data = pd.read_csv(input_csv, header=None, names=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    trans_threshold = 0.001
    rot_threshold = 0.01

    last_pose = np.array(data.iloc[0, 1:])
    for i in range(1, len(data)):
        pose = np.array(data.iloc[i, 1:])
        trans = np.linalg.norm(pose[:3] - last_pose[:3])
        rot = np.linalg.norm(np.array(quat2euler(pose[3:])) - np.array(quat2euler(last_pose[3:])))
        if trans > trans_threshold or rot > rot_threshold:
            start_time = data.iloc[i, 0]
            break
        last_pose = pose

    print('Trajectory file: ', input_csv)
    print('Start time: ', start_time)

    return start_time
