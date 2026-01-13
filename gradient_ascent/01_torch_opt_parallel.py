import torch
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

# load external file
import mirror_simulator

def run_simulation(p_set):
    # エラーハンドリングを追加してプロセスが止まらないようにする
    try:
        return mirror_simulator.simulator(p_set[0], p_set[1], p_set[2])
    except Exception as e:
        print(f"Error in process: {e}")
        return 0.0

if __name__ == "__main__":
    # 1. 浮動小数点数で初期化 (重要)
    params = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
    
    # 2. 最初は SGD + Momentum で挙動を安定させるのがおすすめ
    # lr は 0.5 だと大きすぎる可能性があるため、まずは 0.01 などから試す
    optimizer = torch.optim.SGD([params], lr=0.01, momentum=0.9)
    
    h = 1.0

    # 3. Executorをループの外で定義 (重要: epoch2以降の低速化対策)
    with ProcessPoolExecutor(max_workers=4) as executor:
        
        for step in range(100):
            start = time.time()
            optimizer.zero_grad()
            
            current_p = params.detach().numpy().tolist()
            
            # パラメータセットの作成 [s0用, x+h用, y+h用, yaw+h用]
            tasks = [current_p]
            for i in range(3):
                p_tmp = current_p.copy()
                p_tmp[i] += h
                tasks.append(p_tmp)
            
            # 並列実行
            results = list(executor.map(run_simulation, tasks))
            
            s0 = results[0]
            s_ups = results[1:]
            
            # 4. 勾配計算
            grads = []
            for s_up in s_ups:
                g = (s_up - s0) / h
                grads.append(g)
            
            # 勾配を手動セット（最大化のため -1 を掛ける）
            raw_grad = torch.tensor(grads, dtype=torch.float32) * -1.0
            print(f"  Raw gradients: {raw_grad.numpy()}")
            
            # --- 5. 異常な更新を防ぐための修正 ---
            # 勾配が大きすぎる場合、正規化（クリッピング）する
            # これをしないと params が一瞬で万単位に飛びます
            grad_norm = torch.norm(raw_grad)
            print(f"  Gradient norm: {grad_norm.item():.4f}")
            if grad_norm > 100.0:  # 閾値は適宜調整
                raw_grad = (raw_grad / grad_norm) * 100.0
            
            params.grad = raw_grad
            
            # 更新
            optimizer.step()
            
            # 値が飛びすぎないよう Clamp する（必要に応じて）
            with torch.no_grad():
                # 例: 座標x, y は -50~50, yawは -360~360 の範囲に収める
                params[0:2].clamp_(-50.0, 50.0)
                params[2].clamp_(-360.0, 360.0)
            
            print(f"Step {step}: Params {params.detach().numpy()}, Score {s0}")
            print(f"Iteration time: {time.time() - start:.2f} sec")