import numpy as np
import pandas as pd
import time, json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import coord_trans
import mirror_simulator
import test_func
import point_wise_smvs
import frame_wise_smvs

def get_frame_wise_smvs(points, num_iteration=1, sample_rate=0.5, scale_translation=0.05):

    # (1) Point-wise SMVS calculation
    coordinate, score = point_wise_smvs.calc_pointwise_smvs(points, num_iteration, sample_rate, scale_translation)
    # score (Point-wise SMVSの値域は0から1)

    # (2) Cartesian to Polar conversion
    r , theta = frame_wise_smvs.cartesian2polar(coordinate[:, 0], coordinate[:, 1])

    # (3) Frame-wise score aggregation
    list_angle, list_score = frame_wise_smvs.count_eigen_score(theta, score, 5)

    # (4) Global score calculation
    smvs = frame_wise_smvs.global_score_polar(np.array(list_score), threshold=12)
                    
    # 入力点群数に合わせる
    smvs_normalized = smvs * (points.shape[0]/coordinate.shape[0])

    return smvs_normalized

def binary_to_xyz(binary):
    """Livox custom binary to XYZ numpy array"""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def simulator():

    # --- 設定読み込み ---
    with open('conditions.json', 'r') as f: conditions = json.load(f)

    # --- Config ---
    bag_path = Path(conditions['main']['bag_path'])
    gt_path = Path(conditions['main']['gt_path'])
    map_path = Path(conditions['main']['map_path'])
    lidar_topic_in = conditions['main']['lidar_topic']
    topic_length, lidar_freq = conditions['lidar']['topic_length'], conditions['lidar']['frequency']

    # Point-wise SMVS related params
    num_iteration = 1
    sample_rate = 0.5
    scale_translation = 0.05

    # --- Load Data ---
    num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)
    #map_points_np = load_files.load_pcdfile(map_path)

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC)
    cnt = 0

    # smvs_record
    cnt_record = []
    smvs_record = []

    max_frame_seq = max(num_gt)
    min_frame_seq = min(num_gt)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == lidar_topic_in]
        msg_iter = reader.messages(connections=connections)

        for i, (connection, timestamp, rawdata) in enumerate(msg_iter):
            start = time.time()
            if connection.topic == lidar_topic_in:
                if cnt > max_frame_seq:
                    break
                elif cnt < min_frame_seq:
                    cnt += 1
                    continue
                else:

                    # (1) Deserialize
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    iteration = int(msg.data.shape[0]/topic_length)
                    bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                    lx, ly, lz = binary_to_xyz(bin_points)

                    ly *= -1
                    lz *= -1

                    # (2) Local -> World (GTを使用してシミュレーション用の点群を生成)
                    local_points = np.vstack((lx, ly, lz)).T
                    
                    # 入力点群数に合わせる
                    smvs_normalized = get_frame_wise_smvs(local_points, num_iteration, sample_rate, scale_translation)

                    cnt_record.append(cnt)
                    smvs_record.append(smvs_normalized)
                    print(f"Frame {cnt}: Frame-wise SMVS score: {smvs_normalized:.4f}")

                    cnt += 1
    
    return cnt_record, smvs_record

def write_temp_file(cnt, smvs):
    # cnt, smvs はそれぞれ同長のリストを想定
    try:
        df = pd.DataFrame({'frame': cnt, 'smvs': smvs})
    except Exception:
        # fallback: convert to list explicitly
        df = pd.DataFrame({'frame': list(cnt), 'smvs': list(smvs)})

    out_path = Path('temp_smvs.csv')
    df.to_csv(out_path, mode='a', header=not out_path.exists(), index=False)

if __name__ == "__main__":

    cnt_record, smvs_record = simulator() 
    write_temp_file(cnt_record, smvs_record)