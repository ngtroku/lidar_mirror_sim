import open3d as o3d
import numpy as np
from typing import Tuple

def draw_mirror(center, width, height, rotation_angle=0): # rotation in degrees
    mirror_size = [width, 0.05, height] 
    rotation_angle = np.deg2rad(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])
    obox = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, mirror_size)
    obox.color = (0.0, 0.0, 0.0) 
    return obox

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


def reflection_sim(points, sensor_pos, sensor_ori, mirror_center, mirror_width, mirror_height, R):
    """
    仮想センサ法を用いて鏡面反射による鏡像点群を生成します。
    
    :param points: 世界座標系の全点群 (N, 3)
    :param sensor_pos: センサの位置 [x, y, z]
    :param sensor_ori: センサの姿勢クオータニオン [w, x, y, z]
    :param mirror_center: 鏡の中心 [x, y, z]
    :param mirror_width: 鏡の横幅
    :param mirror_height: 鏡の縦幅
    :param R: 鏡の回転行列 (3x3)
    :return: 鏡像点群 (M, 3), 反射源となった実体点群 (M, 3)
    """
    # 1. 鏡の法線ベクトルと平面の定義
    # 鏡のローカル座標でY軸正の方向を表面(法線)と仮定
    normal_local = np.array([0, 1, 0])
    normal_world = R @ normal_local
    C = np.array(mirror_center)
    S = np.array(sensor_pos)

    # 2. 仮想センサ位置 (S_v) の計算
    # センサを鏡面に対して対称移動させる
    dist_s = np.dot(S - C, normal_world)
    S_v = S - 2 * dist_s * normal_world

    # 3. 反射源の抽出 (仮想センサから見て、鏡の枠内にある点を特定)
    # 仮想センサから各点へのベクトル
    O_v = S_v
    ray_directions = points - O_v
    
    # 鏡面との交差判定: t = (C - O_v)・n / (ray・n)
    numerator = np.dot(C - O_v, normal_world)
    denominator = np.dot(ray_directions, normal_world)
    
    # 分母が0に近い(面と平行な光線)を除外
    valid_denom = np.abs(denominator) > 1e-6
    t = np.zeros(len(points))
    t[valid_denom] = numerator / denominator[valid_denom]
    
    # 交点が仮想センサと実体点Pの間にある (0 < t < 1) かつ、センサの「前」にある点のみ対象
    # (鏡の枠を通して点を見ている条件)
    mask = (t > 0) & (t < 1.0)
    
    # 交点 I の世界座標
    I_world = O_v + t[:, np.newaxis] * ray_directions
    
    # 4. 鏡の枠内(境界)チェック
    # 交点を鏡のローカル座標に変換
    I_local = (R.T @ (I_world - C).T).T
    
    # 鏡のローカル座標系: x=横幅方向, z=高さ方向 と仮定
    half_w = mirror_width / 2.0
    half_h = mirror_height / 2.0
    
    in_boundary = (I_local[:, 0] >= -half_w) & (I_local[:, 0] <= half_w) & \
                  (I_local[:, 2] >= -half_h) & (I_local[:, 2] <= half_h)
    
    final_mask = mask & in_boundary
    P_source = points[final_mask]

    # 5. 鏡像 (P_virtual) の生成
    # 反射源 P_source を鏡面に対して反転させる
    # P' = P - 2 * ((P - C)・n) * n
    if len(P_source) > 0:
        dist_p = np.sum((P_source - C) * normal_world, axis=1)
        P_virtual = P_source - 2 * dist_p[:, np.newaxis] * normal_world
    else:
        P_virtual = np.empty((0, 3))

    return P_virtual, P_source

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