import numpy as np
import open3d as o3d
import time, subprocess, signal, json
from pathlib import Path
import optuna
import matplotlib.pyplot as plt

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import mirror_simulation
import coord_trans
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

def world_to_local(points_world, sensor_pos, sensor_quat):
    """
    世界座標系の点群をセンサーローカル座標系に戻す
    P_local = (P_world - T) @ R
    """
    if len(points_world) == 0:
        return np.empty((0, 3))
        
    R = o3d.geometry.get_rotation_matrix_from_quaternion(sensor_quat)
    # 平行移動の逆
    points_shifted = points_world - np.array(sensor_pos)
    # 回転の逆 (Rは直交行列なので転置が逆行列、ここでは right multiply なので R をそのまま掛ける)
    # local @ R.T = rotated -> rotated @ R = local
    points_local = points_shifted @ R
    
    return points_local

def create_pointcloud2(points, seq, stamp, frame_id, typestore):
    """
    numpy配列(N,3)からsensor_msgs/PointCloud2メッセージを作成する
    """
    # データのバイト列変換 (float32, Little Endian)
    # points は (N, 3) の float32
    blob = points.astype(np.float32).tobytes()
    data_array = np.frombuffer(blob, dtype=np.uint8)
    
    # PointFieldの定義
    PointField = typestore.types['sensor_msgs/msg/PointField']
    fields = [
        PointField(name='x', offset=0, datatype=7, count=1), # 7=FLOAT32
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
    ]

    # Headerの作成
    Header = typestore.types['std_msgs/msg/Header']
    # Timestampの調整 (rosbagsのtimestampはナノ秒int)
    sec = int(stamp // 1_000_000_000)
    nanosec = int(stamp % 1_000_000_000)
    # rosbagsのバージョンによってTime型の扱いが異なるため、単純なオブジェクト構築を使用
    Timestamp = typestore.types['builtin_interfaces/msg/Time']
    ros_time = Timestamp(sec=sec, nanosec=nanosec)
    header = Header(seq=seq, stamp=ros_time, frame_id=frame_id)

    # PointCloud2の構築
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=12, # 4 bytes * 3
        row_step=12 * points.shape[0],
        data=data_array,
        is_dense=True
    )
    return msg

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
        # 例: range=30, pos=10 -> offset=10
        offset = cycle_pos
    elif cycle_pos < 3 * swing_range:
        # Phase 2: 上限 -> 下限 (下降)
        # 例: range=30, pos=40 -> offset = 2*30 - 40 = 20
        # 例: range=30, pos=80 -> offset = 2*30 - 80 = -20
        offset = 2 * swing_range - cycle_pos
    else:
        # Phase 3: 下限 -> 中心 (上昇)
        # 例: range=30, pos=100 -> offset = 100 - 4*30 = -20
        offset = cycle_pos - 4 * swing_range
        
    return base_yaw + offset

def generate_rosbag(param_x=8.0, param_y=0.0, param_yaw_center=0.0, param_swing_speed=0.0, param_swing_range=0.0): # シミュレーションした点群をrosbagに変換
    # --- Load External Configs ---
    # シミュレーション条件の読み込み
    with open('conditions.json', 'r') as f:
        conditions = json.load(f)
    
    # 出力パスなどの基本設定読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)

    # --- Config from JSON ---
    # Main settings
    bag_path = Path(conditions['main']['bag_path']) #
    output_bag_path = Path(config['main']['input_file']) # シミュレーション済みバッグの出力先
    
    lidar_topic_in = conditions['main']['lidar_topic'] #
    lidar_topic_out = conditions['main']['lidar_topic']
    imu_topic = conditions['main']['imu_topic'] #

    # Mirror Config
    #mirror_center = conditions['mirror']['center'] #
    mirror_center = [param_x, param_y, 0.4]
    mirror_width = conditions['mirror']['width'] #
    mirror_height = conditions['mirror']['height'] #

    #mirror_yaw_base = conditions['mirror']['yaw_base'] #
    mirror_yaw_base = param_yaw_center
    #swing_speed = conditions['mirror']['swing_speed'] #
    swing_speed = param_swing_speed
    #swing_range = conditions['mirror']['swing_range'] #
    swing_range = param_swing_range
    
    # LiDAR Config
    FOV_H = conditions['lidar']['fov_h'] #
    FOV_V = conditions['lidar']['fov_v'] #
    topic_length = conditions['lidar']['topic_length'] #
    lidar_freq = conditions['lidar']['frequency'] #

    # --- Remove existing output file if exists ---
    if output_bag_path.exists():
        print(f"Removing existing output file: {output_bag_path}")
        try:
            output_bag_path.unlink()
        except PermissionError:
            print(f"Error: Could not delete {output_bag_path}. Is it open in another program?")
            exit(1)

    # --- Load Data ---
    print("Loading trajectory and map...")
    gt_pose = Path(conditions['main']['gt_path']) #
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_pose)

    map_pcd_path = Path(conditions['main']['map_path']) #
    map_points_np = load_files.load_pcdfile(str(map_pcd_path))

    # --- Setup KDTree ---
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points_np)
    #print("Building KDTree for occlusion check...")
    map_tree = o3d.geometry.KDTreeFlann(map_pcd)

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC)
    cnt = 0 

    #print(f"Processing bag: {bag_path} -> {output_bag_path}")

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        with Writer(output_bag_path) as writer:
            
            # IMU connections setup
            imu_connections = [x for x in reader.connections if x.topic == imu_topic]
            if imu_connections:
                imu_conn_in = imu_connections[0]
                imu_conn_out = writer.add_connection(imu_topic, imu_conn_in.msgtype, typestore=typestore)
            
            # LiDAR connection setup (sensor_msgs/PointCloud2)
            lidar_conn_out = writer.add_connection(lidar_topic_out, 'sensor_msgs/msg/PointCloud2', typestore=typestore)

            connections = [x for x in reader.connections if x.topic == lidar_topic_in or x.topic == imu_topic]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                if connection.topic == imu_topic:
                    writer.write(imu_conn_out, timestamp, rawdata)

                elif connection.topic == lidar_topic_in:
                    if cnt >= len(gt_x):
                        break

                    msg = reader.deserialize(rawdata, connection.msgtype)
                    iteration = int(msg.data.shape[0]/topic_length)
                    bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                    lx, ly, lz = binary_to_xyz(bin_points)
                    
                    local_points = np.vstack((lx, ly, lz)).T
                    sensor_pos = [gt_x[cnt], gt_y[cnt], gt_z[cnt]]
                    sensor_quat = [gt_qw[cnt], gt_qx[cnt], gt_qy[cnt], gt_qz[cnt]]
                    
                    wx, wy, wz = coord_trans.local_to_world(local_points, sensor_pos, sensor_quat)
                    lidar_points_world = np.vstack((wx, wy, wz)).T

                    # 2. シミュレーション処理
                    is_reflected = mirror_simulation.faster_check_intersection(
                        lidar_points_world, mirror_center, mirror_width, mirror_height, mirror_yaw_base, sensor_pos
                    )
                    P_visible = lidar_points_world[~is_reflected]

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

                    simulated_points_world = np.vstack((P_visible, P_virtual_fov))
                    simulated_points_local = world_to_local(simulated_points_world, sensor_pos, sensor_quat)

                    frame_id = msg.header.frame_id if hasattr(msg, 'header') else "livox_frame"
                    
                    out_msg = create_pointcloud2(simulated_points_local, cnt, timestamp, frame_id, typestore)
                    serialized_msg = typestore.serialize_ros1(out_msg, lidar_conn_out.msgtype)
                    writer.write(lidar_conn_out, timestamp, serialized_msg)
                    
                    cnt += 1

def run_slam():
    # 実行するコマンド
    command = ["roslaunch", "slamspoof", "mirror_sim_kiss.launch"]
    print("SLAMを開始します。終了するまで待機中...")
    
    try:
        # subprocess.run はプロセスが終了するまで次の行へ進みません
        # check=True を付けると、コマンドがエラーで終了した際に例外を発生させます
        result = subprocess.run(command, check=True)
        
        print(f"SLAMが正常に終了しました（終了コード: {result.returncode}）")
        
    except subprocess.CalledProcessError as e:
        print(f"SLAMの実行中にエラーが発生しました: {e}")
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。")

def objective(trial): # 鏡シミュレーションからAPEによる評価まで行う
    ground_truth = "/home/rokuto/lidar_mirror_sim/kiss_benign.txt"
    estimated_traj = "/home/rokuto/lidar_mirror_sim/estimated_traj/temp.txt"

    mirror_center_x = trial.suggest_float('mirror_x', 7.5, 7.5)
    mirror_center_y = trial.suggest_float('mirror_y', -3.5, -3.5)
    mirror_orientation = trial.suggest_float('mirror_orientation_yaw', 105, 105) #鏡の向き
    swing_speed = trial.suggest_float('mirror_swing_speed', 0, 20.0) # 鏡を振る速さ(deg/s)
    swing_range = trial.suggest_float('mirror_swing_range', 90, 120) # 鏡を振る範囲(deg)

    generate_rosbag(param_x=mirror_center_x, param_y=mirror_center_y, param_yaw_center=mirror_orientation, param_swing_speed=swing_speed, param_swing_range=swing_range) # rosbag作成
    run_slam() # SLAM実行
    APE = error_estimate.evo_eval_result(estimated_traj, ground_truth) # 目的関数
    return APE 

if __name__ == "__main__":
    start_time = time.time()
    
    # 最適化設定
    optimization = optuna.create_study(direction='maximize')
    
    print("\n" + "="*60)
    print("Start Multi-Parameter Optimization (Position, Orientation, Swing)")
    print("="*60)
    
    # 最適化実行
    optimization.optimize(objective, n_trials=100)

    # -------------------------------------------------------------------------
    # 1. 結果サマリー表示
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print(f"Best APE Value: {optimization.best_value:.4f} m")
    print("-" * 60)
    for key, value in optimization.best_params.items():
        print(f"  {key:<25}: {value:.4f}")
    print("="*60 + "\n")

# -------------------------------------------------------------------------
    # 2. 履歴テーブル表示
    # -------------------------------------------------------------------------
    print("--- Detailed Optimization History ---")
    # ヘッダーをパラメータ名に合わせる
    header = f"{'Trial':<5} | {'APE (Error)':<12} | {'X':<7} | {'Y':<7} | {'Yaw':<7} | {'Speed':<7} | {'Range':<7}"
    print(header)
    print("-" * len(header))
    
    res_x, res_y, res_vals = [], [], []

    for trial in optimization.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            p = trial.params
            val = trial.value
            
            # objective内の trial.suggest_float と名前を完全に一致させる
            # (mirror_pos_x ではなく mirror_x)
            print(f"{trial.number:<5} | {val:<12.4f} | "
                  f"{p['mirror_x']:<7.2f} | {p['mirror_y']:<7.2f} | "
                  f"{p['mirror_orientation_yaw']:<7.1f} | "
                  f"{p['mirror_swing_speed']:<7.1f} | "
                  f"{p['mirror_swing_range']:<7.1f}")
            
            res_x.append(p['mirror_x'])
            res_y.append(p['mirror_y'])
            res_vals.append(val)

    print(f"\nTotal optimization time: {time.time() - start_time:.2f} sec")

    # -------------------------------------------------------------------------
    # 3. グラフ表示 (RuntimeError対策版)
    # -------------------------------------------------------------------------
    res_spd = []
    for trial in optimization.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            res_spd.append(trial.params['mirror_swing_speed'])

    # layout='constrained' でレイアウト崩れを防止
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')

    # --- ax1: 試行回数 vs 誤差 (変更なし) ---
    ax1.plot(range(len(res_vals)), res_vals, marker='o', markersize=4, linestyle='-', alpha=0.4, color='tab:blue')
    ax1.axhline(y=max(res_vals), color='r', linestyle='--', label=f'Max Error: {max(res_vals):.3f}')
    ax1.set_title('Objective Value History (Error Trace)')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('APE (Mean Error [m])')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- ax2: 試行回数 vs 鏡の振る速度 (Swing Speed) ---
    # c=res_vals により、エラーが大きいほど赤く表示
    sc = ax2.scatter(range(len(res_spd)), res_spd, c=res_vals, cmap='jet', s=40, edgecolors='k', alpha=0.7)
    
    # カラーバーの追加
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label('APE (Error [m])')

    # 最良試行（最大エラー）時の速度を強調
    best_trial_idx = np.argmax(res_vals)
    ax2.scatter(best_trial_idx, res_spd[best_trial_idx], color='white', marker='*', s=150, edgecolors='black', label='Best Speed')

    ax2.set_title('Mirror Swing Speed Transition')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Mirror Swing Speed [deg/s]')
    
    # 速度の探索範囲 (0 ~ 20) に合わせて y軸を調整
    ax2.set_ylim(-1, 21) 
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    print("\nDisplaying analysis plots (Speed Transition)...")
    plt.show()