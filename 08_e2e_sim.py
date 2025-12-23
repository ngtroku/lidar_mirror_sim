import numpy as np
import open3d as o3d
import time, json
from pathlib import Path
import matplotlib.pyplot as plt

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
# 1. Helper Functions
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

# -----------------------------------------------------------------------------
# 2. Main Process
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Config ---
    bag_path = Path("./12_17_hap_00.bag")
    
    lidar_topic_in = "/livox/lidar"
    imu_topic = "/livox/imu"

    # Mirror Config
    mirror_center = [8.0, 0, 0.4]
    mirror_width = 1.0
    mirror_height = 0.4
    mirror_yaw_base = 121.7 # base yaw angle (degrees)
    swing_speed = 5.09  # degrees per second
    swing_range =  107.77 # degrees(±)
    
    # LiDAR Config
    FOV_H = 120.0
    FOV_V = 25.0
    topic_length = 18 
    lidar_freq = 10.0 # Hz

    # Load external config 
    with open('config.json', 'r') as f:
        config = json.load(f)

    # For Scan Matching
    source_points = None
    target_points = None
    global_transform = np.identity(4) # Estimate Pose
    estimate_x, estimate_y, estimate_z = [0], [0], [0]

    # --- Load Data ---
    print("Loading trajectory and map...")
    gt_pose = Path("./traj_lidar.txt") # Ground Truth pose
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_pose)

    map_pcd_path = Path("./12_23_map.pcd") 
    map_points_np = load_files.load_pcdfile(map_pcd_path)

    # --- Setup Map KDTree (For ray casting) ---
    print("Building KDTree for occlusion check...")
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points_np)
    map_tree = o3d.geometry.KDTreeFlann(map_pcd) # KDTree for fast search

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC) 
    cnt = 0 # of frames
    start_time = time.time() 

    print(f"Processing bag: {bag_path} (Read-only Simulation)")

    # Only Reader
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        
        # メッセージループ
        connections = [x for x in reader.connections if x.topic == lidar_topic_in or x.topic == imu_topic]
        total_msgs = len(list(reader.messages(connections=connections)))
        print(f"Total messages to process: {total_msgs}")

        # 再度イテレータを作成
        msg_iter = reader.messages(connections=connections)

        for i, (connection, timestamp, rawdata) in enumerate(msg_iter):
            
            # --- IMU: Skip (no writing) ---
            if connection.topic == imu_topic:
                pass

            # --- LiDAR: Simulation Process Only ---
            elif connection.topic == lidar_topic_in:
                if cnt >= len(gt_x):
                    continue 

                # 1. デシリアライズ & 点群抽出
                msg = reader.deserialize(rawdata, connection.msgtype)
                iteration = int(msg.data.shape[0]/topic_length)
                bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                lx, ly, lz = binary_to_xyz(bin_points)
                
                # Local -> World
                local_points = np.vstack((lx, ly, lz)).T
                sensor_pos = [gt_x[cnt], gt_y[cnt], gt_z[cnt]]
                sensor_quat = [gt_qw[cnt], gt_qx[cnt], gt_qy[cnt], gt_qz[cnt]]
                
                wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat)
                lidar_points_world = np.vstack((wx, wy, wz)).T

                # 2. シミュレーション処理
                # (a) 鏡に当たった点(Occluded)を除去
                is_reflected = mirror_simulation.check_intersection(
                    lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos
                )
                P_visible = lidar_points_world[~is_reflected]

                # (b) 光線チェック & 鏡像生成
                is_mirror_visible_los = mirror_simulation.check_line_of_sight(
                    map_tree, sensor_pos, mirror_center, step=0.2, radius=0.15
                )

                P_virtual_fov = np.empty((0, 3))
                if is_mirror_visible_los:

                    mirror_yaw = decide_mirror_yaw_triangular(mirror_yaw_base, swing_range, swing_speed, cnt / lidar_freq)

                    yaw_rad = np.deg2rad(mirror_yaw)
                    Rz = np.array([
                        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
                        [0, 0, 1]
                    ])
                    P_virtual_raw, _ = mirror_simulation.reflection_sim(
                        map_points_np, sensor_pos, sensor_quat, 
                        mirror_center, mirror_width, mirror_height, Rz
                    )
                    P_virtual_fov = filter_by_fov(
                        P_virtual_raw, sensor_pos, sensor_quat, fov_h=FOV_H, fov_v=FOV_V
                    )

                # (c) 点群の合成 (Visible + Virtual)
                # 書き出しはしないが、計算負荷を見るために合成処理までは行う
                simulated_points_world = np.vstack((P_visible, P_virtual_fov))
                
                # 3. World -> Sensor Local 座標変換
                simulated_points_local = coord_trans.world_to_local(simulated_points_world, sensor_pos, sensor_quat)

                # --- Registration ---
                if source_points is None and target_points is None: # initial setting (Frame1) 
                    target_points = simulated_points_local

                elif source_points is None: # 1st registration (Frame2)
                    source_points = simulated_points_local
                    GICP_result = registration.registration_main(target_points, source_points, config)

                    global_transform = global_transform @ GICP_result.T_target_source # Current Pose
                    print(f"x:{global_transform[0, 3]:.4f} y:{global_transform[1, 3]:.4f} z:{global_transform[2, 3]:.4f}")

                    estimate_x.append(global_transform[0, 3])
                    estimate_y.append(global_transform[1, 3])
                    estimate_z.append(global_transform[2, 3])
                
                else:
                    target_points = source_points
                    source_points = simulated_points_local
                    GICP_result = registration.registration_main(target_points, source_points, config)

                    global_transform = global_transform @ GICP_result.T_target_source
                    print(f"x:{global_transform[0, 3]:.4f} y:{global_transform[1, 3]:.4f} z:{global_transform[2, 3]:.4f}")

                    estimate_x.append(global_transform[0, 3])
                    estimate_y.append(global_transform[1, 3])
                    estimate_z.append(global_transform[2, 3])

                cnt += 1
            
            # 進捗表示
            if i % 100 == 0:
                print(f"Processed {i} messages... (LiDAR Frames: {cnt})")

    elapsed = time.time() - start_time
    print(f"Finished simulation. Time elapsed: {elapsed:.2f} seconds.")

    ground_truth_trajectory = np.vstack((gt_x, gt_y, gt_z)).T
    estimated_trajectory = np.vstack((estimate_x, estimate_y, estimate_z)).T

    errors = error_estimate.calc_trans_error(ground_truth_trajectory, estimated_trajectory)
    print(f"Mean Error:{np.mean(errors):.3f}m Std:{np.std(errors):.3f}m")

    plt.plot(gt_x, gt_y, color="blue", label="Ground Truth")
    plt.plot(estimate_x, estimate_y, color="red", label="Mirror Attacked")
    plt.legend()
    plt.show()
