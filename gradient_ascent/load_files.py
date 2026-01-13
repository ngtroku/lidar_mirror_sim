
import csv, re
import open3d as o3d
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

def load_pcdfile(filepath, flip=False):
    """
    PCDファイルを読み込み、numpy配列として返す。
    flip=True の場合、Z軸（高さ）を反転させる。
    """
    # Open3Dで読み込み（Pathオブジェクトにも対応できるようstr型に変換）
    pcd = o3d.io.read_point_cloud(str(filepath))

    # numpy配列に変換
    points = np.asarray(pcd.points)

    # 上下反転オプション
    if flip:
        # Z軸(インデックス2)の符号を反転
        # points[:, 2] *= -1  # 単純な反転
        
        # もし「座標の原点は変えずに、その場での上下向きだけ変えたい」場合は
        # 以下のように最大値と最小値を入れ替える方法もあります
        # z_max = np.max(points[:, 2])
        # points[:, 2] = z_max - points[:, 2]
        
        points[:, 2] = -points[:, 2]
        print(f"Point cloud flipped vertically (Z-axis inverted).")

    return points

def load_benign_pose(filepath):

    df = pd.read_csv(filepath, delim_whitespace=True, names=('timestamp','x','y','z','qx','qy','qz','qw'))
    
    x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()
    qx, qy, qz, qw = df['qx'].to_numpy(), df['qy'].to_numpy(), df['qz'].to_numpy(), df['qw'].to_numpy()

    return x, y, z, qw, qx, qy, qz # クオータニオンはw,x,y,zの順

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

def load_smvs(filepath):
    df = pd.read_csv(filepath)
    smvs_values = df['smvs'].to_numpy()
    return smvs_values

def save_results(optimization):
    # 1. 現在の日時を取得 (例: 20240325_153045)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. ファイル名を生成 (optimization_results_日時.csv)
    filename = f"optimization_results_{timestamp}.csv"
    optimization_csv = Path("./") / filename
    
    # Optunaの全試行データをpandas DataFrameとして取得
    df_results = optimization.trials_dataframe()
    
    # CSVとして保存
    df_results.to_csv(optimization_csv, index=False)
    
    print(f"Optimization history saved to {optimization_csv}")




    