import numpy as np
import open3d as o3d
import time, json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import optuna

# rosbags libraries
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# import external functions
import load_files
import coord_trans
import calc_smvs
import objective_function
import test_func

def get_mirror_orientation_yaw(mirror_pos_x, mirror_pos_y):
    # load config 
    with open('conditions.json', 'r') as f:
        config = json.load(f)
    
    gt_path = Path(config['main']['gt_path'])
    num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)
    
    # search nearest point
    gt_coords_2d = np.vstack((gt_x, gt_y)).T
    tree = KDTree(gt_coords_2d)
    query_point = [mirror_pos_x, mirror_pos_y]
    dist, idx = tree.query(query_point)

    # find yaw orientation
    target_x, target_y = gt_x[idx], gt_y[idx]
    dx, dy = target_x - mirror_pos_x, target_y - mirror_pos_y
    yaw_rad = np.arctan2(dy, dx)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg

def objective(trial):
    # 1. 探索するパラメータの範囲を指定

    mirror_x = trial.suggest_float('mirror_pos_x', -20, 60)
    #mirror_x = trial.suggest_float('mirror_pos_x', 10, 40)
    mirror_y = trial.suggest_float('mirror_pos_y', 0, 120)
    #mirror_y = trial.suggest_float('mirror_pos_y', 80, 120)
    nearest_dist = test_func.measure_nearest_dist(mirror_x, mirror_y)

    param_swing_speed = trial.suggest_float('param_swing_speed', 0.0, 15.0)
    #mirror_orientation = trial.suggest_float('mirror_orientation_yaw', -180, 180)

    if nearest_dist >= 5.0 and nearest_dist <= 15.0:
        print(f"Trial {trial.number} meets condition: nearest distance {nearest_dist:.2f}m.\n")
        try:
            score = objective_function.simulator(mirror_x, mirror_y, 
                param_yaw_center=get_mirror_orientation_yaw(mirror_x, mirror_y), param_swing_speed=param_swing_speed)
            return score
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0
    else:
        print(f"Trial {trial.number} skipped: nearest distance {nearest_dist:.2f}m out of range.\n")
        return 0.0

if __name__ == "__main__":

    # debug variable
    is_not_generate_tempfile = True
    
    if is_not_generate_tempfile:
        print("Skipping temporary SMVS file generation.")

    else:
        #1. smvs一時ファイル作成
        cnt_record, smvs_record = calc_smvs.simulator()
        calc_smvs.write_temp_file(cnt_record, smvs_record)
        print("Temporary SMVS file created.")

    # 2. 最適化の設定
    # 計算時間がかかるため、多次元探索に強い TPESampler を明示
    study = optuna.create_study(
        direction='maximize',      # スコアを最大化する
        sampler=optuna.samplers.TPESampler(n_startup_trials=200) # 最初の25回はランダム探索
    )

    # 3. 最適化の実行
    # n_trials: 合計何回シミュレーションを回すか
    # 1回200秒なら、30回で約1.6時間、100回で約5.5時間です
    study.optimize(objective, n_trials=750)
    load_files.save_results(study)

    # 4. 結果の表示
    print("Optimization finished.")
    print(f"Best score: {study.best_value}")
    print(f"Best parameters: {study.best_params}")

    # 5. 後処理
    # --- プロット: 試行で提案された候補点をXY平面に表示 ---

    # load ground truth
    gt_path = Path('estimated_pose.csv')
    num_gt, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz = load_files.load_benign_pose_csv(gt_path)

    try:
        xs = []
        ys = []
        vals = []
        for t in study.trials:
            if 'mirror_pos_x' in t.params and 'mirror_pos_y' in t.params:
                xs.append(t.params['mirror_pos_x'])
                ys.append(t.params['mirror_pos_y'])
                vals.append(t.value if t.value is not None else np.nan)

        if len(xs) > 0:
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(xs, ys, c=vals, cmap='viridis', edgecolor='k', s=60)
            plt.colorbar(sc, label='objective value')
            # best point
            best = study.best_params
            if 'mirror_pos_x' in best and 'mirror_pos_y' in best:
                plt.scatter([best['mirror_pos_x']], [best['mirror_pos_y']], c='red', marker='*', s=200, label='best')
                plt.legend()
            
            plt.plot(gt_x, gt_y, c='gray', marker='.', linestyle='None', alpha=0.3, label='Ground Truth Path')
            plt.xlabel('mirror_pos_x')
            plt.ylabel('mirror_pos_y')
            plt.title('Optuna candidate points (XY plane)')
            out_fig = Path('opt_candidates_xy.png')
            plt.tight_layout()
            plt.savefig(out_fig)
            try:
                plt.show()
            except Exception:
                pass
            print(f"Saved candidate plot to {out_fig}")
    except Exception as e:
        print(f"Plotting candidates failed: {e}")

    # 一時ファイル削除
    if is_not_generate_tempfile:
        pass
    else:
        temp_path = Path('temp_smvs.csv')
        if temp_path.exists():
            temp_path.unlink()
            print("Temporary SMVS file deleted.")
        else:
            print("No temporary SMVS file found to delete.")
    