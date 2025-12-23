import open3d as o3d
import numpy as np
import time, json
from typing import Tuple
from pathlib import Path

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# load external files
import load_files
import coord_trans

def binary_to_xyz(binary):
    """Livox custom binary to XYZ numpy array"""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

# check intersection (old)
def check_intersection(point_cloud_data: np.ndarray, center: list, width: float, height: float, yaw_angle: float, sensor_pos: list) -> np.ndarray:
    """
    点群データ内の各点が、センサー位置と点群を結ぶ直線上で鏡と交差するかを判定します。
    """
    N = point_cloud_data.shape[0]
    is_reflected = np.zeros(N, dtype=bool)
    
    # センサー位置を新しい座標Oとして定義
    O = np.array(sensor_pos) 
    C = np.array(center)

    yaw_rad = np.deg2rad(yaw_angle)
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    normal_local = np.array([0, 1, 0])
    normal_world = Rz @ normal_local
    plane_point = C

    for i in range(N):
        P = point_cloud_data[i]
        
        # 光線の方向ベクトルは P - O
        ray_direction = P - O

        denominator = np.dot(ray_direction, normal_world)

        if np.abs(denominator) < 1e-6:
            continue
        
        # 分子: (平面上の点 - 光線始点 O) と法線ベクトルの内積
        numerator = np.dot(plane_point - O, normal_world) 
        t = numerator / denominator

        # 光線が鏡を通り越すか、センサー O の後ろにある場合はスキップ
        if t <= 0.0 or t > 1.0: 
            continue
        
        # 交点 I の世界座標: O + t * ray_direction
        I_world = O + t * ray_direction

        # 境界チェック (長方形領域内にあるか)
        I_local_shifted = I_world - C
        I_local = Rz.T @ I_local_shifted
        
        x_local, y_local, z_local = I_local
        
        half_width = width / 2.0
        half_height = height / 2.0
        
        is_inside_width = (-half_width <= x_local <= half_width)
        is_inside_height = (-half_height <= z_local <= half_height)

        if is_inside_width and is_inside_height:
            is_reflected[i] = True

    return is_reflected

def faster_check_intersection(point_cloud_data: np.ndarray, center: list, width: float, height: float, yaw_angle: float, sensor_pos: list) -> np.ndarray:

    # 1. 定数の準備
    O = np.array(sensor_pos)
    C = np.array(center)
    half_width = width / 2.0
    half_height = height / 2.0

    yaw_rad = np.deg2rad(yaw_angle)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    
    # ローカルから世界座標への回転行列 Rz
    Rz = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0, 0, 1]
    ])
    
    # 鏡の法線ベクトル（世界座標系）
    normal_world = Rz @ np.array([0, 1, 0])
    
    # 2. 光線の計算
    # 各点 P に対して ray_direction = P - O
    ray_directions = point_cloud_data - O  # (N, 3)
    
    # 3. 交点パラメータ t の一括計算
    # 線の式: L = O + t * ray_direction
    # 面の式: (L - C)・normal_world = 0
    # t = ((C - O)・normal_world) / (ray_direction・normal_world)
    
    # 分母 (N,)
    denominators = ray_directions @ normal_world
    
    # 分子 (スカラー)
    numerator = np.dot(C - O, normal_world)
    
    # ゼロ除算を避けるためのマスク
    valid_mask = np.abs(denominators) > 1e-6
    
    # t を計算 (N,)
    # 有効な分母以外は一旦 0 にして計算し、後でマスクをかける
    t = np.zeros_like(denominators)
    t[valid_mask] = numerator / denominators[valid_mask]
    
    # 4. 範囲チェック (t の条件)
    # 0 < t <= 1.0 : センサーと点 P の間に鏡がある
    t_mask = valid_mask & (t > 0.0) & (t <= 1.0)
    
    # 5. 交点の世界座標を計算
    # I = O + t * ray_direction (対象となる点のみ)
    # 効率化のため、t_mask が True の点だけ計算する
    indices = np.where(t_mask)[0]
    if len(indices) == 0:
        return np.zeros(point_cloud_data.shape[0], dtype=bool)
    
    I_world = O + t[indices, np.newaxis] * ray_directions[indices]
    
    # 6. 鏡のローカル座標系への変換と境界チェック
    # I_local = Rz^T @ (I_world - C)
    I_local_shifted = I_world - C
    # 行列演算で一括変換 (I_local_shifted @ Rz は 各行ベクトル v に v @ Rz を適用することと同等)
    I_local = I_local_shifted @ Rz
    
    x_local = I_local[:, 0]
    z_local = I_local[:, 2]
    
    # 境界チェック
    inside_mask = (np.abs(x_local) <= half_width) & (np.abs(z_local) <= half_height)
    
    # 最終的な結果配列の作成
    is_reflected = np.zeros(point_cloud_data.shape[0], dtype=bool)
    is_reflected[indices[inside_mask]] = True
    
    return is_reflected

def simulator():

    # Load registration config
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load simulation conditions
    with open('conditions.json', 'r') as f:
        conditions = json.load(f)

    # --- Parse Variables from Conditions ---

    # Main settings

    bag_path = Path(conditions['main']['bag_path'])
    gt_path = Path(conditions['main']['gt_path'])
    map_path = Path(conditions['main']['map_path'])
    lidar_topic_in = conditions['main']['lidar_topic']
    imu_topic = conditions['main']['imu_topic']

    # Mirror settings
    mirror_center = conditions['mirror']['center']

    #mirror_center = [param_x, param_y, 0.4]
    mirror_width = conditions['mirror']['width']
    mirror_height = conditions['mirror']['height']
    mirror_yaw_base = conditions['mirror']['yaw_base']

    #swing_speed = param_swing_speed
    swing_speed = 5.0 # deg/s
    #swing_range = param_swing_range
    swing_range = 105 # degree

    # LiDAR settings
    FOV_H = conditions['lidar']['fov_h']
    FOV_V = conditions['lidar']['fov_v']
    topic_length = conditions['lidar']['topic_length']
    lidar_freq = conditions['lidar']['frequency']

    # --- Initialize Variables for Process ---
    source_points = None
    target_points = None
    global_transform = np.identity(4) # Estimated Pose
    estimate_x, estimate_y, estimate_z = [0.0], [0.0], [0.0] # Trajectory history

    # --- Load Data ---

    print(f"Loading Ground Truth from {gt_path}...")
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_path)
    print(f"Loading Map from {map_path}...")
    map_points_np = load_files.load_pcdfile(map_path)

    # --- Setup Map KDTree (For ray casting) ---

    print("Building KDTree for occlusion check...")
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points_np)
    map_tree = o3d.geometry.KDTreeFlann(map_pcd)

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC)
    cnt = 0 # frame counter
    start_time = time.time()
    print(f"Processing bag: {bag_path} (Read-only Simulation)")

    # processing time recording
    processing_time = []
    processing_time_faster = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:

        connections = [x for x in reader.connections if x.topic == lidar_topic_in or x.topic == imu_topic]
        total_msgs = len(list(reader.messages(connections=connections)))
        print(f"Total messages to process: {total_msgs}")

        msg_iter = reader.messages(connections=connections)

        for i, (connection, timestamp, rawdata) in enumerate(msg_iter):
            # --- IMU: Skip ---
            if connection.topic == imu_topic:
                pass
            # --- LiDAR: Simulation & Registration ---
            elif connection.topic == lidar_topic_in:

                if cnt >= len(gt_x):
                    continue

                # 1. Deserialize & Extract Points
                msg = reader.deserialize(rawdata, connection.msgtype)
                iteration = int(msg.data.shape[0]/topic_length)
                bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                lx, ly, lz = binary_to_xyz(bin_points) # Assuming this function exists

                # Local -> World (using GT)

                local_points = np.vstack((lx, ly, lz)).T
                sensor_pos = [gt_x[cnt], gt_y[cnt], gt_z[cnt]]
                sensor_quat = [gt_qw[cnt], gt_qx[cnt], gt_qy[cnt], gt_qz[cnt]]
                wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat)

                lidar_points_world = np.vstack((wx, wy, wz)).T

                # 2. Simulation Process
                # (a) Occlusion Check
                t1_start = time.time()
                is_reflected = check_intersection(
                lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos)
                t1_end = time.time()

                t2_start = time.time()
                is_reflected_vectorized = faster_check_intersection(lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos)
                t2_end = time.time()

                print(f"Processing time original:{(t1_end-t1_start):.5f}sec")
                print(f"Processing time faster:{(t2_end-t2_start):.5f}sec")

                processing_time.append((t1_end-t1_start))
                processing_time_faster.append((t2_end-t2_start))
    
    return processing_time, processing_time_faster

if __name__ == "__main__":
    processing_time, processing_time_faster = simulator()
    print(f"Original Processing time:{(np.mean(processing_time)):.5f} ± {(np.std(processing_time)):.5f}sec")
    print(f"Faster Processing time:{(np.mean(processing_time_faster)):.5f} ± {(np.std(processing_time_faster)):.5f}sec")