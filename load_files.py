
import csv, re
import open3d as o3d
import numpy as np
import pandas as pd

def load_pcdfile(filepath):
    # Load a point cloud from a PCD file and return as an Open3D PointCloud object.
    pcd = o3d.io.read_point_cloud(filepath)

    # convert to numpy array for processing
    points = np.asarray(pcd.points)

    return points

def load_benign_pose(filepath):

    df = pd.read_csv(filepath, delim_whitespace=True, names=('timestamp','x','y','z','qx','qy','qz','qw'))
    
    x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()
    qx, qy, qz, qw = df['qx'].to_numpy(), df['qy'].to_numpy(), df['qz'].to_numpy(), df['qw'].to_numpy()

    return x, y, z, qw, qx, qy, qz # クオータニオンはw,x,y,zの順



    