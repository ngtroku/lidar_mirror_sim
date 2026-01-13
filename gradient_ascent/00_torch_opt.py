import torch
import numpy as np
import time

# load external file
import mirror_simulator

if __name__ == "__main__":

    # パラメータ初期値
    x_init, y_init, yaw_init = 0.0, 0.0, 0.0
    params = torch.tensor([x_init, y_init, yaw_init], requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.5)
    h = 0.5 # 数値微分のステップ

    for step in range(100):
        start = time.time()
        optimizer.zero_grad()
        
        # 1. 現在のスコア
        s0 = mirror_simulator.simulator(params[0].item(), params[1].item(), params[2].item())
        
        # 2. 各軸の勾配を数値微分で計算
        grads = []

        for i in range(3):
            p_temp = params.clone().detach()
            p_temp[i] += h
            s_up = mirror_simulator.simulator(p_temp[0].item(), p_temp[1].item(), p_temp[2].item())
            grads.append((s_up - s0) / h)
        
        # 3. PyTorchの勾配として手動でセット
        # Adamは「損失の最小化」を想定するため、上昇させたい場合は勾配の符号を反転させる
        params.grad = torch.tensor(grads, dtype=torch.float32) * -1.0
        
        # 4. Adamによる更新
        optimizer.step()
        
        print(f"Step {step}: Params {params.detach().numpy()}, Score {s0}")
        print(f"processing time")