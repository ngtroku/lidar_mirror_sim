import numpy as np
import open3d as o3d
import time, json
from pathlib import Path
import optuna

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import mirror_simulation
import coord_trans
import registration
import error_estimate   

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

def simulator(param_yaw_center, param_swing_speed, param_swing_range):
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
    mirror_width = conditions['mirror']['width']
    mirror_height = conditions['mirror']['height']
    mirror_yaw_base = param_yaw_center
    swing_speed = param_swing_speed
    swing_range = param_swing_range

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
                is_reflected = mirror_simulation.check_intersection(
                    lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos
                )
                P_visible = lidar_points_world[~is_reflected]

                # (b) Mirror Image Generation (Line of Sight Check)
                is_mirror_visible_los = mirror_simulation.check_line_of_sight(
                    map_tree, sensor_pos, mirror_center, step=0.2, radius=0.15
                )

                P_virtual_fov = np.empty((0, 3))
                if is_mirror_visible_los:
                    # Calculate dynamic mirror yaw (Triangular Wave)
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
                    # Assuming filter_by_fov is available
                    P_virtual_fov = filter_by_fov(
                        P_virtual_raw, sensor_pos, sensor_quat, fov_h=FOV_H, fov_v=FOV_V
                    )

                # (c) Merge Points
                simulated_points_world = np.vstack((P_visible, P_virtual_fov))
                
                # 3. World -> Local (for Scan Matching Input)
                simulated_points_local = coord_trans.world_to_local(simulated_points_world, sensor_pos, sensor_quat)

                # --- Registration (Scan Matching) ---
                if source_points is None and target_points is None: # Initial frame
                    target_points = simulated_points_local

                elif source_points is None: # 1st registration
                    source_points = simulated_points_local
                    GICP_result = registration.registration_main(target_points, source_points, config)

                    global_transform = global_transform @ GICP_result.T_target_source
                    
                    estimate_x.append(global_transform[0, 3])
                    estimate_y.append(global_transform[1, 3])
                    estimate_z.append(global_transform[2, 3])
                
                else: # Sequential registration
                    target_points = source_points
                    source_points = simulated_points_local
                    GICP_result = registration.registration_main(target_points, source_points, config)

                    global_transform = global_transform @ GICP_result.T_target_source
                    
                    estimate_x.append(global_transform[0, 3])
                    estimate_y.append(global_transform[1, 3])
                    estimate_z.append(global_transform[2, 3])

                cnt += 1
            
            # Progress log
            if i % 100 == 0:
                print(f"Processed {i} messages... (LiDAR Frames: {cnt})")

    elapsed = time.time() - start_time
    print(f"Finished simulation. Time elapsed: {elapsed:.2f} seconds.")

    # --- Error Evaluation ---
    # Adjust GT length to match estimated length
    min_len = min(len(gt_x), len(estimate_x))
    ground_truth_trajectory = np.vstack((gt_x[:min_len], gt_y[:min_len], gt_z[:min_len])).T
    estimated_trajectory = np.vstack((estimate_x[:min_len], estimate_y[:min_len], estimate_z[:min_len])).T

    # Calculate Error
    errors = error_estimate.calc_trans_error(ground_truth_trajectory, estimated_trajectory)
    print(f"Mean Error: {np.mean(errors):.3f}m, Std: {np.std(errors):.3f}m")

    return np.mean(errors)

def objective(trial):
    mirror_orientation = trial.suggest_float('mirror_orientation_yaw', -180, 180)
    swing_speed = trial.suggest_float('mirror_swing_speed', 0, 20.0)
    swing_range = trial.suggest_float('mirror_swing_range', 0, 90)

    obj_error = simulator(mirror_orientation, swing_speed, swing_range)
    print(f"Mean Error:{obj_error:.3f}m")
    return obj_error

if __name__ == "__main__":
    optimization = optuna.create_study(direction='maximize')
    optimization.optimize(objective, n_trials=3)

    print(f'Best value: {optimization.best_value}')
    print(f'Best param: {optimization.best_params}')
    





