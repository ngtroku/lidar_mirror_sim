import numpy as np
import open3d as o3d
import time
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import coord_trans
import mirror_simulation

def binary_to_xyz(binary):
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def filter_by_fov(points, sensor_pos, sensor_quat, fov_h=120, fov_v=25):
    """
    点群をセンサのFOVでフィルタリングする
    points: (N, 3) 世界座標系
    sensor_pos: [x, y, z]
    sensor_quat: [w, x, y, z]
    fov_h, fov_v: 視野角 (度)
    """
    if len(points) == 0:
        return points

    # 1. World -> Sensor Local 座標変換
    R_sensor = o3d.geometry.get_rotation_matrix_from_quaternion(sensor_quat)
    diff = points - np.array(sensor_pos)
    points_local = diff @ R_sensor # (R * P.T).T と同等

    # 2. 角度計算 (LiDAR座標系: X前方, Y左, Z上)
    x = points_local[:, 0]
    y = points_local[:, 1]
    z = points_local[:, 2]

    # Azimuth (水平角)
    azimuth = np.arctan2(y, x)
    # Elevation (垂直角)
    hypot_xy = np.hypot(x, y)
    elevation = np.arctan2(z, hypot_xy)

    # 3. マスク作成
    fov_h_rad = np.deg2rad(fov_h)
    fov_v_rad = np.deg2rad(fov_v)

    mask = (x > 0) & \
           (np.abs(azimuth) <= fov_h_rad / 2.0) & \
           (np.abs(elevation) <= fov_v_rad / 2.0)

    return points[mask]

# --- 光線チェック関数 (KDTree使用) ---
def check_line_of_sight(pcd_tree, start_pos, end_pos, step=0.2, radius=0.15):
    """
    センサー(start)から鏡(end)への直線上に障害物(地図点群)があるか判定する
    :param pcd_tree: 地図点群のKDTree
    :param start_pos: センサー位置
    :param end_pos: ターゲット(鏡)位置
    :return: True=見える(遮蔽なし), False=見えない(遮蔽あり)
    """
    vec = np.array(end_pos) - np.array(start_pos)
    dist = np.linalg.norm(vec)
    if dist < 1e-3: return True
    
    direction = vec / dist

    # 自分のすぐ近く(0.5m)と、鏡の直前(0.2m)はチェックしない
    current_dist = 0.5 
    target_dist = dist - 0.2

    while current_dist < target_dist:
        check_point = np.array(start_pos) + direction * current_dist
        
        # 半径 radius 内に障害物点群があるか検索
        k, _, _ = pcd_tree.search_radius_vector_3d(check_point, radius)
        
        if k > 0:
            return False # 障害物あり

        current_dist += step

    return True # 障害物なし

if __name__ == "__main__":

    # parameters
    bag_path = Path("./12_17_hap_00.bag")
    point_topic = "/livox/lidar"
    imu_topic = "/livox/imu"

    # mirror setups
    mirror_center = [8.0, 0, 0.4]
    mirror_width = 1.0
    mirror_height = 0.4
    mirror_yaw = -45

    # Livox HAP の FOV (度)
    FOV_H = 120.0
    FOV_V = 25.0

    # --- 1. Ground Truth Pose の読み込み ---
    gt_pose = Path("./traj_lidar.txt")
    gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose(gt_pose)

    # --- 2. Open3D Visualizer のセットアップ ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Sequence Replay", width=1024, height=768)

    # A) 全体マップ
    map_pcd = o3d.geometry.PointCloud()
    map_points_np = load_files.load_pcdfile("./benign_pcd_full.pcd")
    map_pcd.points = o3d.utility.Vector3dVector(map_points_np)
    map_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # 地図は青色
    vis.add_geometry(map_pcd)
    
    # ★追加: KDTreeの構築 (ループ外で1回だけ実行)
    print("Building KDTree for occlusion check...")
    map_tree = o3d.geometry.KDTreeFlann(map_pcd)

    # B) 現在フレーム用
    current_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(current_pcd)

    # C) 鏡像シミュレーション用
    mirror_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(mirror_pcd)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(axis)

    topic_length = 18 
    typestore = get_typestore(Stores.ROS1_NOETIC)
    cnt = 0 

    print("Starting replay...")

    # --- 3. メインループ ---
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == point_topic or x.topic == imu_topic]

        for connection, timestamp, rawdata in reader.messages(connections=connections):

            if connection.topic == point_topic:
                if cnt >= len(gt_x):
                    break

                msg = reader.deserialize(rawdata, connection.msgtype)

                # decode point cloud
                iteration = int(msg.data.shape[0]/topic_length)
                bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
                lx, ly, lz = binary_to_xyz(bin_points)
                local_points = np.vstack((lx, ly, lz)).T

                # Local to World frame
                sensor_position = [gt_x[cnt], gt_y[cnt], gt_z[cnt]]
                sensor_orientation = [gt_qw[cnt], gt_qx[cnt], gt_qy[cnt], gt_qz[cnt]] # [w,x,y,z]
                world_x, world_y, world_z = coord_trans.local_to_world(local_points, sensor_position, sensor_orientation)
                lidar_points_world = np.vstack((world_x, world_y, world_z)).T

                # --- 1. 鏡に当たった点(Occluded)とそれ以外(Visible)を分離 ---
                is_reflected = mirror_simulation.check_intersection(lidar_points_world, mirror_center, 
                                                                    mirror_width, mirror_height, mirror_yaw, sensor_position)
                P_visible = lidar_points_world[~is_reflected]
                # P_occluded = lidar_points_world[is_reflected]    

                # --- 2. ★KDTreeを使った光線チェック (Occlusion Check) ---
                # センサーから鏡が見えているか判定
                is_mirror_visible_los = check_line_of_sight(
                    map_tree, 
                    sensor_position, 
                    mirror_center, 
                    step=0.2, 
                    radius=0.15
                )

                P_virtual_fov = np.empty((0, 3)) # 初期化

                if is_mirror_visible_los:
                    # 鏡が見えている場合のみ、鏡像計算を行う
                    
                    yaw_rad = np.deg2rad(mirror_yaw)
                    Rz = np.array([
                        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
                        [0, 0, 1]
                    ])    
                    
                    P_virtual_raw, _ = mirror_simulation.reflection_sim(map_points_np, sensor_position, 
                                                                        sensor_orientation, mirror_center, mirror_width, mirror_height, Rz)

                    # --- 3. 鏡像を現在のFOVでカットする ---
                    P_virtual_fov = filter_by_fov(P_virtual_raw, sensor_position, sensor_orientation, 
                                                  fov_h=FOV_H, fov_v=FOV_V)
                
                # generate mirror simulated pointcloud (保存などが必要な場合に使用)
                # simulated_points = np.vstack((P_visible, P_virtual_fov))

                # --- Update Visualizer (修正: remove -> add 方式) ---
                # update_geometry は点群サイズ変更に対応できないため、再登録を行う
                
                # A) 現在フレーム (赤)
                vis.remove_geometry(current_pcd, reset_bounding_box=False)
                current_pcd.points = o3d.utility.Vector3dVector(P_visible) 
                current_pcd.paint_uniform_color([1.0, 0.0, 0.0])
                vis.add_geometry(current_pcd, reset_bounding_box=False)

                # B) 鏡像点群 (緑)
                vis.remove_geometry(mirror_pcd, reset_bounding_box=False)
                if len(P_virtual_fov) > 0:
                    mirror_pcd.points = o3d.utility.Vector3dVector(P_virtual_fov)
                    mirror_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                    vis.add_geometry(mirror_pcd, reset_bounding_box=False)

                vis.poll_events()
                vis.update_renderer()

                cnt += 1
                time.sleep(0.02) 

            elif connection.topic == imu_topic:
                pass

    print("Finished.")
    vis.destroy_window()