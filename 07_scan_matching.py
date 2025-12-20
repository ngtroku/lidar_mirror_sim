from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import numpy as np
import json
import matplotlib.pyplot as plt

# import external module
import registration

def binary_to_xyz(binary):
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def main(config):
    bagpath = Path(str(config['main']['input_file']))
    topic_length = int(config['main']['topic_length'])

    source_points = None
    target_points = None

    cnt = 0

    # Estimate Odometry
    global_transform = np.identity(4)
    odom_x, odom_y, odom_z = 0, 0, 0
    traj_x, traj_y, traj_z = [], [], []

    # Create a type store
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # visualize setting
    if config['main']['visualize']:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Estimated Trajectory")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
    
    # Create reader instance
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == config['main']['lidar_topic']]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Binary decoding
            iteration = int(msg.data.shape[0] / topic_length)
            bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length)
            x, y, z = binary_to_xyz(bin_points) 
            
            current_points = np.vstack((x, y, z)).T

            # Registration Process
            if source_points is None and target_points is None: # Initial frame
                target_points = current_points

            elif source_points is None: # Second frame (First registration)
                source_points = current_points
                
                # Registration
                result = registration.registration_main(target_points, source_points, config)
                
                # Update Transform and Trajectory
                global_transform = global_transform @ result.T_target_source
                odom_x = -1 * global_transform[0, 3]
                odom_y = global_transform[1, 3]
                odom_z = global_transform[2, 3]

                traj_x.append(odom_x)
                traj_y.append(odom_y)
                traj_z.append(odom_z)

                print(f"Frame:{cnt} x:{odom_x:.3f}, y:{odom_y:.3f}, z:{odom_z:.3f}")

                # Visualization
                if config['main']['visualize']:
                    ax.plot(traj_x, traj_y, color="blue")
                    plt.pause(0.01)
                
                cnt += 1

            else: # Subsequent frames
                target_points = source_points
                source_points = current_points
                
                # Registration
                result = registration.registration_main(target_points, source_points, config)

                # Update Transform and Trajectory
                global_transform = global_transform @ result.T_target_source
                odom_x = global_transform[0, 3]
                odom_y = global_transform[1, 3]
                odom_z = global_transform[2, 3]

                traj_x.append(odom_x)
                traj_y.append(odom_y)
                traj_z.append(odom_z)

                print(f"Frame:{cnt} x:{odom_x:.3f}, y:{odom_y:.3f}, z:{odom_z:.3f}")

                # Visualization
                if config['main']['visualize']:
                    ax.plot(traj_x, traj_y, color="blue")
                    plt.pause(0.01)

                cnt += 1

    if config['main']['visualize']:
        plt.show()

if __name__ == '__main__':
    # load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    main(config)