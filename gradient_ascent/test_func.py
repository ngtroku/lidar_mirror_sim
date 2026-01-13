import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import KDTree
import json

def point_stats(points):

    distances = np.linalg.norm(points, axis=1) # 原点からの距離
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    return mean_dist, std_dist

def load_benign_pose_csv(filepath):
    # pose_inW.csv はカンマ区切りでヘッダーがあるため、デフォルトの pd.read_csv で読み込みます
    df = pd.read_csv(filepath)
    #print(df)

    num = df['num'].to_numpy()
    
    # 座標データを取り出し
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    
    # クオータニオンデータを取り出し
    qx = df['qx'].to_numpy()
    qy = df['qy'].to_numpy()
    qz = df['qz'].to_numpy()
    qw = df['qw'].to_numpy()

    # 指定の戻り値順: x, y, z, qw, qx, qy, qz
    return num, x, y, z, qw, qx, qy, qz

def measure_nearest_dist(x, y):
    # 1. コンフィグの読み込み
    with open('conditions.json', 'r') as f:
        config = json.load(f)

    num, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_benign_pose_csv(config['main']['gt_path'])

    # 3. 2次元座標の配列を作成 (N, 2)
    gt_points = np.vstack((gt_x, gt_y)).T

    # --- 高速化のためのKDTree利用 ---
    # 大規模な軌跡データの場合、全ての点との距離を計算するより高速です
    tree = KDTree(gt_points)
    
    # query(点, k=近傍数) で距離(d)とインデックス(i)を返す
    dist, idx = tree.query([x, y])

    return dist