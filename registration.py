
import numpy as np
import small_gicp
from scipy.spatial.transform import Rotation as R

def registration_main(target, source, config):

    # settings
    registration_type = str(config['registration']['registration_type'])
    downsampling_resolution = float(config['registration']['downsampling_resolution'])
    max_correspondence_distance = float(config['registration']['max_correspondence_distance'])
    num_threads = int(config['registration']['num_threads'])

    result = small_gicp.align(target, source, registration_type=registration_type, 
    downsampling_resolution=downsampling_resolution, max_correspondence_distance=max_correspondence_distance,
    num_threads=num_threads)
    
    return result

def transform_matrix_2_quaternion(transform_matrix):

    rot_matrix = transform_matrix[:3, :3]
    r = R.from_matrix(rot_matrix)
    
    # Quat (x, y, z, w)
    return r.as_quat()

def transform_matrix_2_euler(transform_matrix):
    
    rot_matrix = transform_matrix[:3, :3]
    r = R.from_matrix(rot_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return roll, pitch, yaw

def transform_quat_2_euler(quat):
    r = R.from_quat(quat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return roll, pitch, yaw

def get_cov(Hessian):
    eigen_value, eigen_vector = np.linalg.eig(Hessian)
    axis_width, axis_height = eigen_value[0], eigen_value[1]
    xy_cov = eigen_vector[0:2, 0:2]
    return axis_width, axis_height, xy_cov