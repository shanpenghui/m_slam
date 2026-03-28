import pandas as pd
import numpy as np

def sync_pose_pair(intput_A, intput_B, timediff_A, timediff_B, output_A, output_B):

    data_A = pd.read_csv(intput_A, header=None, names=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    data_B = pd.read_csv(intput_B, header=None, names=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    data_A['t'] -= timediff_A
    data_B['t'] -= timediff_B

    time_threshold = 0.01

    A_result = []
    B_result = []
    for i in range(len(data_A)):
        time_A = data_A.iloc[i, 0]
        diff = np.abs(data_B['t'] - time_A)
        if diff.min() < time_threshold:
            index = diff.idxmin()
            A_result.append(data_A.iloc[i, 1:])
            B_result.append(data_B.iloc[index, 1:])

    pd.DataFrame(A_result).to_csv(output_A, header=None, index=None)
    pd.DataFrame(B_result).to_csv(output_B, header=None, index=None)