
import numpy as np
import small_gicp
import sys
import time

def random_sampling(array, sample_rate): # random sampling

    num_sample = int(array.shape[0] * sample_rate)

    indices_1 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array1 = array[indices_1]

    indices_2 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array2 = array[indices_2]

    return sampled_array1, sampled_array2

def points_noise(array, scale_translation):
    rng = np.random.default_rng()

    # x, y, z 軸方向の各点に正規分布からサンプリングされたノイズを定義
    noise_x = rng.normal(0, scale_translation, array.shape[0])
    noise_y = rng.normal(0, scale_translation, array.shape[0])
    noise_z = rng.normal(0, scale_translation, array.shape[0])

    # ノイズを与える
    array_noised = array.copy()
    array_noised[:, 0] += noise_x
    array_noised[:, 1] += noise_y
    array_noised[:, 2] += noise_z

    return array_noised

def calc_factor(source_points, target_points):
    source, source_tree = small_gicp.preprocess_points(source_points, downsampling_resolution=0.3)
    target, target_tree = small_gicp.preprocess_points(target_points, downsampling_resolution=0.3)

    result = small_gicp.align(target, source, target_tree)
    result = small_gicp.align(target, source, target_tree, result.T_target_source)

    factors = [small_gicp.GICPFactor()]
    rejector = small_gicp.DistanceRejector()

    # initialize
    sum_H = np.zeros((6, 6))
    sum_b = np.zeros(6)
    sum_e = 0.0

    # 全体ヘッセの最小固有ベクトルを求める（対称行列であれば eigh が速い）
    try:
        eigen_value, eigen_vector = np.linalg.eigh(result.H)
        # eigh は昇順で固有値を返すので最小は index 0
        global_min_vector = eigen_vector[:, 0]
    except Exception:
        eigen_value, eigen_vector = np.linalg.eig(result.H)
        global_min_vector = eigen_vector[:, np.argmin(eigen_value)]

    # 事前確保: 成功する点の最大数は source.size() と見積もる
    n_pts = source.size()
    xyz_arr = np.empty((n_pts, 3), dtype=float)
    cov_vals = np.empty((n_pts,), dtype=float)
    filled = 0

    for i in range(n_pts):
        succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
        if not succ:
            continue

        # 各点ヘッセの固有値・固有ベクトル（対称なら eigh で高速化）
        try:
            point_eigen_value, point_eigen_vector = np.linalg.eigh(H)
            # 最大固有値に対応するベクトルは最後の列
            local_max_vector = point_eigen_vector[:, -1]
        except Exception:
            point_eigen_value, point_eigen_vector = np.linalg.eig(H)
            local_max_vector = point_eigen_vector[:, np.argmax(point_eigen_value)]

        # グローバルで最も拘束が弱い方向との内積 (正規化済みのベクトル同士を想定)
        naiseki = np.dot(global_min_vector.real, local_max_vector.real)
        cov_vals[filled] = (naiseki + 1) / 2.0
        xyz_arr[filled] = source.points()[i, 0:3]
        filled += 1

        sum_H += H
        sum_b += b
        sum_e += e

    if filled == 0:
        return np.empty((0, 3)), np.empty((0,))

    return xyz_arr[:filled], cov_vals[:filled]

def calc_pointwise_smvs(array, num_iteration, sample_rate, scale_translation):
    counter = 1
    pc1, pc2 = random_sampling(array, sample_rate)

    while counter <= int(num_iteration):

        if counter == 1:
            #print("iteration:{}/{}".format(counter, num_iteration))
            #t1 = time.time()
            source, target = pc1, points_noise(pc2, scale_translation)
            #print(f"  points_noise time: {time.time() - t1:.4f} sec")

            #t2 = time.time()
            coordinate_origin, dot_eigen_value_origin = calc_factor(source, target)
            #print(f"  calc_factor time: {time.time() - t2:.4f} sec")
            counter += 1

        else:
            #print("iteration:{}/{}".format(counter, num_iteration))
            #t1 = time.time()
            source, target = pc1, points_noise(pc2, scale_translation)
            #print(f"  points_noise time: {time.time() - t1:.4f} sec")

            #t2 = time.time()
            coordinate, dot_eigen_value = calc_factor(source, target)
            #print(f"  calc_factor time: {time.time() - t2:.4f} sec")

            coordinate_origin = np.concatenate([coordinate_origin, coordinate])
            dot_eigen_value_origin = np.concatenate([dot_eigen_value_origin, dot_eigen_value])
            counter += 1
    
    return coordinate_origin, dot_eigen_value_origin