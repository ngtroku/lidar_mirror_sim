import numpy as np
import open3d as o3d
import time, json
from pathlib import Path
import optuna
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import mirror_simulation
import coord_trans
import registration
import error_estimate   

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def binary_to_xyz(binary):
    """Livox custom binary to XYZ numpy array"""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def filter_by_fov(points, sensor_pos, sensor_quat, fov_h=120, fov_v=25):
    if len(points) == 0:
        return points

    # World -> Sensor Local
    R_sensor = o3d.geometry.get_rotation_matrix_from_quaternion(sensor_quat)
    diff = points - np.array(sensor_pos)
    points_local = diff @ R_sensor 

    x = points_local[:, 0]
    y = points_local[:, 1]
    z = points_local[:, 2]

    azimuth = np.arctan2(y, x)
    hypot_xy = np.hypot(x, y)
    elevation = np.arctan2(z, hypot_xy)

    fov_h_rad = np.deg2rad(fov_h)
    fov_v_rad = np.deg2rad(fov_v)

    mask = (x > 0) & \
           (np.abs(azimuth) <= fov_h_rad / 2.0) & \
           (np.abs(elevation) <= fov_v_rad / 2.0)

    return points[mask]

def decide_mirror_yaw_triangular(base_yaw, swing_range, rotation_speed, current_time):
    if swing_range == 0 or rotation_speed == 0:
        return base_yaw

    # 1サイクル（中心→最大→最小→中心）の移動総距離は range * 4
    cycle_distance = 4 * swing_range
    
    # 現在までの総移動距離
    total_distance = rotation_speed * current_time
    
    # 現在のサイクル内の位置 (0 ～ 4*range)
    cycle_pos = total_distance % cycle_distance
    
    offset = 0.0
    
    if cycle_pos < swing_range:
        # Phase 1: 中心 -> 上限 (上昇)
        offset = cycle_pos
    elif cycle_pos < 3 * swing_range:
        # Phase 2: 上限 -> 下限 (下降)
        offset = 2 * swing_range - cycle_pos
    else:
        # Phase 3: 下限 -> 中心 (上昇)
        offset = cycle_pos - 4 * swing_range
        
    return base_yaw + offset

def get_mirror_orientation_yaw(mirror_pos_x, mirror_pos_y):
    # load config 
    with open('conditions.json', 'r') as f:
        config = json.load(f)
    
    gt_path = Path(config['main']['gt_path'])
    num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)
    
    # search nearest point
    gt_coords_2d = np.vstack((gt_x, gt_y)).T
    tree = KDTree(gt_coords_2d)
    query_point = [mirror_pos_x, mirror_pos_y]
    dist, idx = tree.query(query_point)

    # find yaw orientation
    target_x, target_y = gt_x[idx], gt_y[idx]
    dx, dy = target_x - mirror_pos_x, target_y - mirror_pos_y
    yaw_rad = np.arctan2(dy, dx)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg

# -----------------------------------------------------------------------------
# Simulator Function (計測機能付き)
# -----------------------------------------------------------------------------
def simulator(param_x, param_y, param_yaw_center, param_swing_speed, param_swing_range, mirror_width=2.0, mirror_height=2.0):

    # --- 設定読み込み ---
    with open('conditions.json', 'r') as f: conditions = json.load(f)
    with open('config.json', 'r') as f: config = json.load(f)

    # --- Config ---
    bag_path = Path(conditions['main']['bag_path'])
    gt_path = Path(conditions['main']['gt_path'])
    map_path = Path(conditions['main']['map_path'])
    lidar_topic_in = conditions['main']['lidar_topic']
    
    mirror_center = [param_x, param_y, 0.0]
    mirror_width, mirror_height = mirror_width, mirror_height
    mirror_yaw_base, swing_speed, swing_range = param_yaw_center, param_swing_speed, param_swing_range
    FOV_H, FOV_V = conditions['lidar']['fov_h'], conditions['lidar']['fov_v']
    topic_length, lidar_freq = conditions['lidar']['topic_length'], conditions['lidar']['frequency']

    # --- Load Data ---
    num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)
    map_points_np = load_files.load_pcdfile(map_path)

    # --- Setup Map KDTree ---
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points_np)
    map_tree = o3d.geometry.KDTreeFlann(map_pcd)

    # --- 2. 初期姿勢(クオータニオン含む)の設定 ---
    # GTの最初のフレーム番号を取得
    cnt = int(num_gt[0])
    
    # GTの1フレーム目から初期姿勢行列を作成
    initial_pos = np.array([gt_x[0], gt_y[0], gt_z[0]])
    initial_quat = [gt_qx[0], gt_qy[0], gt_qz[0], gt_qw[0]] # scipy準拠 [x,y,z,w]
    r_matrix = R.from_quat(initial_quat).as_matrix()

    global_transform = np.identity(4)
    global_transform[:3, :3] = r_matrix
    global_transform[:3, 3] = initial_pos

    # 保存用リスト: [num, x, y, z, qx, qy, qz, qw]
    results = [[cnt, gt_x[0], gt_y[0], gt_z[0], gt_qx[0], gt_qy[0], gt_qz[0], gt_qw[0]]]

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC)
    load_num = 0
    source_points, target_points = None, None
    
    # 軌跡描画用のリスト（CSV保存用 results と別に管理すると便利）
    estimate_x, estimate_y, estimate_z = [gt_x[0]], [gt_y[0]], [gt_z[0]]

    max_frame_seq = max(num_gt)
    min_frame_seq = min(num_gt)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == lidar_topic_in]
        msg_iter = reader.messages(connections=connections)

        for i, (connection, timestamp, rawdata) in enumerate(msg_iter):

            if connection.topic == lidar_topic_in:
                if cnt > max_frame_seq:
                    break
                elif cnt < min_frame_seq:
                    cnt += 1
                    continue
                else:
                    # (1) Deserialize
                    t_step = time.time()
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    iteration = int(msg.data.shape[0]/topic_length)
                    bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                    lx, ly, lz = binary_to_xyz(bin_points)

                    ly *= -1
                    lz *= -1

                    # (2) Local -> World (GTを使用してシミュレーション用の点群を生成)
                    t_step = time.time()
                    local_points = np.vstack((lx, ly, lz)).T
                    sensor_pos = [gt_x[load_num], gt_y[load_num], gt_z[load_num]]
                    sensor_quat = [gt_qw[load_num], gt_qx[load_num], gt_qy[load_num], gt_qz[load_num]]
                    wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat)
                    lidar_points_world = np.vstack((wx, wy, wz)).T

                    # (3)-(5) Simulation 処理 (中略: 元のコードのまま)
                    t_step = time.time()
                    is_reflected = mirror_simulation.faster_check_intersection(
                        lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos)
                    P_visible = lidar_points_world[~is_reflected]

                    t_step = time.time()
                    is_mirror_visible_los = mirror_simulation.check_line_of_sight(
                        map_tree, sensor_pos, mirror_center, step=0.2, radius=0.15)
                    P_virtual_fov = np.empty((0, 3))
                    if is_mirror_visible_los:
                        mirror_yaw = decide_mirror_yaw_triangular(mirror_yaw_base, swing_range, swing_speed, cnt / lidar_freq)
                        yaw_rad = np.deg2rad(mirror_yaw)
                        Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
                        P_virtual_raw, _ = mirror_simulation.reflection_sim(
                            map_points_np, sensor_pos, sensor_quat, mirror_center, mirror_width, mirror_height, Rz)
                        P_virtual_fov = filter_by_fov(P_virtual_raw, sensor_pos, sensor_quat, fov_h=FOV_H, fov_v=FOV_V)

                    simulated_points_world = np.vstack((P_visible, P_virtual_fov))
                    simulated_points_local = coord_trans.world_to_local(simulated_points_world, sensor_pos, sensor_quat)

                    # (6) Registration (推定姿勢の更新)
                    t_step = time.time()
                    if target_points is None:
                        target_points = simulated_points_local
                    else:
                        source_points = simulated_points_local
                        GICP_result = registration.registration_main(target_points, source_points, config)
                        
                        # 初期姿勢行列に対して相対変化を掛け合わせて更新
                        global_transform = global_transform @ GICP_result.T_target_source
                        
                        # 姿勢の抽出
                        x, y, z = global_transform[0, 3], global_transform[1, 3], global_transform[2, 3]
                        quat = R.from_matrix(global_transform[:3, :3]).as_quat() # [qx, qy, qz, qw]

                        # 記録
                        estimate_x.append(x)
                        estimate_y.append(y)
                        estimate_z.append(z)
                        results.append([cnt, x, y, z, quat[0], quat[1], quat[2], quat[3]])
                        
                        target_points = source_points

                    cnt += 1
                    load_num += 1
        
    return estimate_x, estimate_y, estimate_z

if __name__ == "__main__":

    #mirror_x, mirror_y = 31.6, 101.3
    mirror_x, mirror_y =  -8.43, 19.1

    #mirror_yaw_center = -7.5
    #mirror_yaw_center = 0.0

    # decide mirror yaw based on trajectory
    mirror_yaw_center = get_mirror_orientation_yaw(mirror_x, mirror_y)

    swing_speed = 7.19
    #swing_speed = 5.04
    swing_range = 120.0

    mirror_width, mirror_height = 2.0, 0.8
    estimate_x, estimate_y, estimate_z = simulator(mirror_x, mirror_y, mirror_yaw_center,
                                    swing_speed, swing_range, mirror_width=mirror_width, mirror_height=mirror_height)

    with open('conditions.json', 'r') as f: conditions = json.load(f)
    gt_path = Path(conditions['main']['gt_path'])
    num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)

    # (2) Mirror Placement Map
    plt.scatter(gt_x, gt_y, color="gray", label="Ground Truth Trajectory", s=1)
    plt.plot(estimate_x, estimate_y, color="blue", label="Estimated Trajectory", linewidth=1)

    # Mirror center and orientation (mirror_yaw_center を可視化)
    plt.scatter(mirror_x, mirror_y, color="red", label="Mirror Center", s=50, zorder=5)
    yaw_rad = np.deg2rad(mirror_yaw_center)
    # 矢印の長さは鏡幅か最小1mを基準に調整
    arrow_len = max(1.0, min(mirror_width * 1.5, 5.0))
    dx = arrow_len * np.cos(yaw_rad)
    dy = arrow_len * np.sin(yaw_rad)
    plt.arrow(mirror_x, mirror_y, dx, dy, head_width=0.4, head_length=0.6, fc='red', ec='red', linewidth=2, zorder=6)
    plt.text(mirror_x + dx * 0.1, mirror_y + dy * 0.1, f"yaw={mirror_yaw_center}°", color='red')

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.legend()
    plt.show()