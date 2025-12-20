import numpy as np
import open3d as o3d

# load external files
import coord_trans
import load_files
import mirror_simulation

# Test single frame

if __name__ == "__main__":

    # files
    map_path = "./benign_pcd.pcd" # pre-built map
    scan_path = "./frame_0441.pcd" # LiDAR scan

    # Sensor setups
    sensor_position = np.array([7.365650, -10.348300, 0.0])  # LiDAR position [x, y, z] in meters
    sensor_orientation = np.array([-0.471389, 0.026494, 0.014743, 0.881404])  # LiDAR orientation as quaternion [w, x, y, z]
    geometry = []

    # mirror setups
    mirror_center = [6.5, -12, 0.4]
    mirror_width = 1.0
    mirror_height = 0.8
    mirror_yaw = -10

    # load map
    map_points = load_files.load_pcdfile(map_path)

    # draw mirror & LiDAR sensor
    mirror = mirror_simulation.draw_mirror(mirror_center, mirror_width, mirror_height, rotation_angle=mirror_yaw)
    geometry.append(mirror)

    sensor_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sensor_marker.paint_uniform_color([0.0, 0.0, 0.0])
    sensor_marker.translate(sensor_position)
    geometry.append(sensor_marker)

    # visualize map
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(map_points)
    pcd_map.paint_uniform_color([0.0, 0.0, 1.0]) # gray
    geometry.append(pcd_map)
    
    # load LiDAR scan
    lidar_points = load_files.load_pcdfile(scan_path)
    # transform LiDAR points to world coordinate
    world_x, world_y, world_z = coord_trans.local_to_world(lidar_points, sensor_position, sensor_orientation)
    lidar_points_world = np.vstack((world_x, world_y, world_z)).T

    # simulate mirror occlusion
    is_reflected = mirror_simulation.check_intersection(lidar_points_world, mirror_center, 
    mirror_width, mirror_height, mirror_yaw, sensor_position)
    P_visible = lidar_points_world[~is_reflected]
    P_occluded = lidar_points_world[is_reflected]

    # simulate mirror reflection 
    yaw_rad = np.deg2rad(mirror_yaw)
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    print("Generating virtual image from visible points...")
    P_virtual_simulated, _ = mirror_simulation.reflection_sim(
        map_points,   # 赤い点群を反射させる
        sensor_position, 
        sensor_orientation,
        mirror_center, 
        mirror_width, 
        mirror_height, 
        Rz
    )

    # 鏡像を考慮した点群を生成
    print("Generating virtual image from occluded points...")
    mirror_simulated_points = np.vstack((P_visible, P_virtual_simulated))

    # visualize visible, occluded points & virtual image points
    pcd_visible = o3d.geometry.PointCloud()
    pcd_visible.points = o3d.utility.Vector3dVector(P_visible)
    pcd_visible.paint_uniform_color([0.0, 1.0, 0.4]) # green
    #geometry.append(pcd_visible)

    pcd_occluded = o3d.geometry.PointCloud()
    pcd_occluded.points = o3d.utility.Vector3dVector(P_occluded)
    pcd_occluded.paint_uniform_color([1.0, 0.0, 0.0]) # red
    #geometry.append(pcd_occluded)

    # visualize simulated points
    pcd_mirror_simulated = o3d.geometry.PointCloud()
    pcd_mirror_simulated.points = o3d.utility.Vector3dVector(mirror_simulated_points)
    pcd_mirror_simulated.paint_uniform_color([0.0, 1.0, 1.0]) # cyan
    #geometry.append(pcd_mirror_simulated)

    pcd_virtual = o3d.geometry.PointCloud()
    pcd_virtual.points = o3d.utility.Vector3dVector(P_virtual_simulated)
    pcd_virtual.paint_uniform_color([1.0, 0.0, 0.0]) # Orange
    geometry.append(pcd_virtual)

    # visualize LiDAR scan
    #pcd_lidar = o3d.geometry.PointCloud()
    #pcd_lidar.points = o3d.utility.Vector3dVector(lidar_points_world)
    #pcd_lidar.paint_uniform_color([1.0, 0.0, 0.0]) # red
    #geometry.append(pcd_lidar)

    # visualize all objects
    o3d.visualization.draw_geometries(geometry, window_name="Map Point Cloud")


