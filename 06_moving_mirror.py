import numpy as np
import open3d as o3d
import time
from pathlib import Path

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import mirror_simulation
import coord_trans

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

# -----------------------------------------------------------------------------
# 2. Main Process
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Config ---
    bag_path = Path("./12_17_hap_00.bag")
    output_bag_path = Path("./12_17_hap_simulated.bag") # 出力ファイル名
    
    lidar_topic_in = "/livox/lidar"
    lidar_topic_out = "/livox/lidar" # 同じトピック名にする場合（型は変わります）
    imu_topic = "/livox/imu"

    # Mirror Config
    mirror_center = [8.0, 0, 0.4]
    mirror_width = 1.0
    mirror_height = 0.4
    mirror_yaw_base = -45 # base yaw angle (degrees)
    swing_speed = 5.0  # degrees per second
    swing_range = 30.0 # degrees(±)
    
    # LiDAR Config
    FOV_H = 120.0
    FOV_V = 25.0
    topic_length = 18 
    lidar_freq = 10.0 # Hz

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
    gt_pose = Path("./traj_lidar.txt")
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_pose)

    map_pcd_path = Path("./benign_pcd_full.pcd") # ファイル名を確認してください
    map_points_np = load_files.load_pcdfile(map_pcd_path)

    # --- Open3D Setup ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Simulating & Recording...", width=1024, height=768)

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points_np)
    map_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    vis.add_geometry(map_pcd)

    # ★KDTree構築
    print("Building KDTree for occlusion check...")
    map_tree = o3d.geometry.KDTreeFlann(map_pcd)

    current_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(current_pcd)
    mirror_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(mirror_pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # --- Rosbags Setup ---
    typestore = get_typestore(Stores.ROS1_NOETIC) # ROS1の場合
    
    cnt = 0 

    print(f"Processing bag: {bag_path} -> {output_bag_path}")

    # ReaderとWriterを同時に開く
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        with Writer(output_bag_path) as writer:
            
            # 接続（Connections）の準備
            # Writerに対して「このトピック名で、この型で書きます」と登録する必要がある
            
            # A) IMU: 入力と同じ型を使用 (Copy)
            imu_connections = [x for x in reader.connections if x.topic == imu_topic]
            if imu_connections:
                # 最初の接続情報を使ってWriterに登録
                imu_conn_in = imu_connections[0]
                imu_conn_out = writer.add_connection(imu_topic, imu_conn_in.msgtype, typestore=typestore)
            
            # B) LiDAR: 出力は sensor_msgs/PointCloud2 に変更
            # Livox CustomMsg を編集して再パックするのは困難なため、標準的なPC2に変換します
            lidar_conn_out = writer.add_connection(lidar_topic_out, 'sensor_msgs/msg/PointCloud2', typestore=typestore)

            # メッセージループ
            connections = [x for x in reader.connections if x.topic == lidar_topic_in or x.topic == imu_topic]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                
                # --- IMU: そのまま書き出し ---
                if connection.topic == imu_topic:
                    writer.write(imu_conn_out, timestamp, rawdata)

                # --- LiDAR: 編集して書き出し ---
                elif connection.topic == lidar_topic_in:
                    if cnt >= len(gt_x):
                        break # GT終了

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
                    simulated_points_world = np.vstack((P_visible, P_virtual_fov))
                    
                    # 3. World -> Sensor Local 座標変換 (書き出し用)
                    simulated_points_local = world_to_local(simulated_points_world, sensor_pos, sensor_quat)

                    # 4. メッセージ作成 & 書き出し
                    # frame_id は元のメッセージから取得するか、固定値 (例: "livox_frame")
                    frame_id = msg.header.frame_id if hasattr(msg, 'header') else "livox_frame"
                    
                    out_msg = create_pointcloud2(simulated_points_local, cnt, timestamp, frame_id, typestore)
                    serialized_msg = typestore.serialize_ros1(out_msg, lidar_conn_out.msgtype)
                    writer.write(lidar_conn_out, timestamp, serialized_msg)

                    # 5. 可視化更新
                    vis.remove_geometry(current_pcd, reset_bounding_box=False)
                    current_pcd.points = o3d.utility.Vector3dVector(P_visible) 
                    current_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # 赤色
                    vis.add_geometry(current_pcd, reset_bounding_box=False)

                    vis.remove_geometry(mirror_pcd, reset_bounding_box=False)
                    if len(P_virtual_fov) > 0:
                        mirror_pcd.points = o3d.utility.Vector3dVector(P_virtual_fov)
                        mirror_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # 緑色
                        vis.add_geometry(mirror_pcd, reset_bounding_box=False)

                    vis.poll_events()
                    vis.update_renderer()

                    cnt += 1
                    # 書き出し処理が入るのでsleepは短め、または無しでOK
                    #time.sleep(0.01) 

                else:
                    pass

    print("Finished processing and writing.")
    vis.destroy_window()