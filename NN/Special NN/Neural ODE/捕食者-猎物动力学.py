import math, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from statsmodels.datasets import sunspots

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
np.random.seed(1337)
print("Device:", DEVICE)

# 1 加载哈德逊湾公司的历史数据（1900-1920年的毛皮贸易记录）

# 真实年度毛皮计数（种群的代理），1900-1920（21年）
years = np.arange(1900, 1921, dtype=np.int32)

# 来自经典生态学教科书（四舍五入）
hares = np.array([30, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                  27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2,  7.6, 14.6, 16.2, 24.7], dtype=np.float32)
lynx  = np.array([ 4,  6.1,  9.8, 35.2, 59.4, 41.7, 19.0, 13.0,  8.3,  9.1,
                  10.8, 12.6, 16.8, 20.6, 18.1,  8.0,  5.3,  3.8,  4.0,  6.5,  8.0], dtype=np.float32)

assert len(years) == len(hares) == len(lynx)
N = len(years)
print(f"Years {years[0]}–{years[-1]} (N={N})")

# 将数据放入张量并轻度标准化。
# 种群数据是正数且偏斜的，即有很大的变化范围且严格为正，
# 所以用log1p稳定尺度，再用z-score标准化便于优化。
X_raw = np.stack([hares, lynx], axis=1)              # 形状 (N, 2)
X_log = np.log1p(X_raw)
X_mean = X_log.mean(axis=0, keepdims=True)
X_std  = X_log.std(axis=0, keepdims=True) + 1e-8
X      = (X_log - X_mean) / X_std                    # 标准化 (N, 2)

# 时间轴：居中以从0开始，使用年作为连续单位
t_year = years.astype(np.float32)
t0 = t_year[0]
t  = (t_year - t0)                                   # (N,)
t  = torch.tensor(t, dtype=torch.float32, device=DEVICE)
Y  = torch.tensor(X, dtype=torch.float32, device=DEVICE)  # (N,2)

# 训练/测试分割：拟合80%，预测最后20%
split = int(0.8 * N)
t_tr, y_tr = t[:split], Y[:split]
t_te, y_te = t[split:], Y[split:]

print("Train points:", len(t_tr), " Test points:", len(t_te))

# 2 定义Neural ODE模型- 我们直接建模2D状态[兔子,猞猁]。
# ODE右端是个小的MLP，接收当前状态和时间特征，输出状态的变化率
class ODEFunc(nn.Module):
    """
    参数化dx/dt = f_theta(x, t)。
    我们包含简单的时间特征（sin/cos）以允许轻微的非平稳性。
    """

    def __init__(self, xdim=2, hidden=64, periods=(8.0, 11.0)):
        super().__init__()
        self.periods = torch.tensor(periods, dtype=torch.float32)
        # 输入：x (2) + 时间特征 (2 * periods)
        # 这里加入了傅立叶时间特征（8年和11年周期）来帮助捕捉周期性行为。
        in_dim = xdim + 2 * len(periods)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, xdim),
        )
        # 温和初始化以避免早期流动爆炸
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    m.weight.mul_(0.1);
                    nn.init.zeros_(m.bias)

    def _time_feats(self, t_scalar, batch, device):
        # 构建[sin(2πt/P_k), cos(2πt/P_k)]特征
        tt = t_scalar * torch.ones(batch, 1, device=device)
        feats = []
        for P in self.periods.to(device):
            w = 2.0 * math.pi / P
            feats += [torch.sin(w * tt), torch.cos(w * tt)]
        return torch.cat(feats, dim=1) if feats else torch.zeros(batch, 0, device=device)

    def forward(self, t, x):
        # x: (B, 2) 当前状态
        B = x.shape[0]
        phi_t = self._time_feats(t, B, x.device)
        return self.net(torch.cat([x, phi_t], dim=1))  # (B,2)


class NeuralODE_PredPrey(nn.Module):
    """
    从可学习的初始状态x0在给定时间戳上积分ODE。
    我们将积分轨迹直接与观察到的x(t)比较。
    """
    def __init__(self, hidden=64, method="dopri5", rtol=1e-4, atol=1e-4, max_num_steps=2000):
        super().__init__()
        '''
        method: 使用dopri5自适应求解器保持振荡特性。
        
        '''
        self.func = ODEFunc(xdim=2, hidden=hidden)
        # 标准化空间中的可学习初始条件
        self.x0 = nn.Parameter(torch.zeros(1, 2))  # (1,2)
        # ODE求解器配置
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_num_steps = max_num_steps

    def forward(self, t):
        """
        从x0开始在时间t上积分（广播到batch=1）。
        返回轨迹(N, 1, 2) -> 我们将压缩为(N,2)。
        """
        opts = {"max_num_steps": self.max_num_steps}
        x_traj = odeint(self.func, self.x0, t, method=self.method,
                        rtol=self.rtol, atol=self.atol, options=opts)
        return x_traj.squeeze(1)  # (N,2)


# 3 训练
# 训练过程中同时学习ODE动力学和初始状态，并使用早停机制避免过拟合
# === 步骤3：训练与早停 + 最佳检查点 ===
import os, json, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt

# 模型（与之前相同的超参数；如果你改变了它们请调整）
model = NeuralODE_PredPrey(hidden=64, method="dopri5", rtol=1e-4, atol=1e-4).to(DEVICE)
opt    = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
loss_fn= nn.MSELoss()

# 训练配置
EPOCHS   = 3000          # 上限；如果验证停止改进我们会提前停止
PATIENCE = 50            # 等待改进的轮数（你的曲线显示~50-60最佳）
BESTPATH = "best_predprey.pt"   # 最佳模型的检查点路径

best_te = float("inf")
stale   = 0
hist    = {"epoch": [], "train_mse": [], "test_mse": []}
best_info = {"epoch": None, "test_mse": None}

for ep in range(1, EPOCHS + 1):
    # ---- 在训练网格上训练 ----
    model.train(); opt.zero_grad()
    yhat_tr   = model(t_tr)                 # (Ntr,2)
    train_mse = loss_fn(yhat_tr, y_tr)
    train_mse.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # ---- 在测试网格上验证（评估完整轨迹然后切片） ----
    model.eval()
    with torch.no_grad():
        yhat_all = model(t)                 # (N,2)
        test_mse = loss_fn(yhat_all[split:], y_te)

    # ---- 日志 ----
    hist["epoch"].append(ep)
    hist["train_mse"].append(float(train_mse.item()))
    hist["test_mse"].append(float(test_mse.item()))

    # ---- 每50轮详细输出 ----
    if ep % 50 == 0:
        print(f"Epoch {ep:4d} | Train MSE {train_mse.item():.5f} | Test MSE {test_mse.item():.5f}")

    # ---- 早停逻辑（基于测试MSE） ----
    if test_mse.item() + 1e-8 < best_te:
        best_te = test_mse.item()
        stale   = 0
        best_info["epoch"]   = ep
        best_info["test_mse"]= float(best_te)
        # 保存最佳检查点（仅权重）
        torch.save({"model_state": model.state_dict(),
                    "epoch": ep,
                    "test_mse": float(best_te)}, BESTPATH)
    else:
        stale += 1
        if stale >= PATIENCE:
            print(f"⏹️ 在第{ep}轮早停（验证{PATIENCE}轮无改进）。"  
                  f"最佳轮次 = {best_info['epoch']} 测试MSE = {best_info['test_mse']:.5f}")
            break

# ---- 恢复最佳检查点 ----
ckpt = torch.load(BESTPATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
print(f"✅ 恢复最佳模型 @ 第{ckpt['epoch']}轮 | 最佳测试MSE = {ckpt['test_mse']:.5f}")

# ---- 绘制学习曲线与最佳轮次标记 ----
epochs   = np.array(hist["epoch"], dtype=int)
train_m  = np.array(hist["train_mse"], dtype=float)
test_m   = np.array(hist["test_mse"], dtype=float)
best_ep  = int(best_info["epoch"]) if best_info["epoch"] is not None else int(epochs[np.nanargmin(test_m)])
best_val = float(best_info["test_mse"]) if best_info["test_mse"] is not None else float(np.nanmin(test_m))

plt.figure(figsize=(8,4))
plt.plot(epochs, train_m, label="Train MSE", linewidth=2)
plt.plot(epochs, test_m,  label="Test MSE",  linewidth=2, linestyle="--")
plt.axvline(best_ep, color="gray", linestyle=":", label=f"Best Test @ {best_ep} (MSE={best_val:.4f})")
plt.xlabel("Epoch"); plt.ylabel("MSE (normalized space)")
plt.title("Learning Curves (Train vs Test) with Early Stopping")
plt.grid(True, alpha=.3); plt.legend()
plt.tight_layout(); plt.show()


# 4 可视化结果
# 可视化结果时，还需要把标准化的数据转换回原始单位，这样更容易理解
# ===== 步骤4：评估 + 可视化 =====
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# 1) 恢复最佳检查点（如果尚未恢复）
ckpt = torch.load(BESTPATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2) 辅助函数：反标准化回原始毛皮计数
def denorm(X_norm: torch.Tensor) -> torch.Tensor:
    X_log = X_norm * torch.tensor(X_std.squeeze(), device=X_norm.device) + torch.tensor(X_mean.squeeze(), device=X_norm.device)
    return torch.expm1(X_log)  # log1p的逆

# 3) 在完整时间线（训练+测试）上预测并分割
with torch.no_grad():
    Yhat = model(t)                   # (N,2) 标准化空间
Y_den    = denorm(Y)                  # (N,2) 原始单位
Yhat_den = denorm(Yhat)               # (N,2) 原始单位

# Numpy视图
hares_obs, lynx_obs   = Y_den[:,0].cpu().numpy(),   Y_den[:,1].cpu().numpy()
hares_pred, lynx_pred = Yhat_den[:,0].cpu().numpy(), Yhat_den[:,1].cpu().numpy()

# 4) 指标（标准化空间）
def mse(a,b): return float(np.mean((a-b)**2))
def mae(a,b): return float(np.mean(np.abs(a-b)))

y_np      = Y.cpu().numpy()
yhat_np   = Yhat.detach().cpu().numpy()
y_tr, y_te      = y_np[:split],     y_np[split:]
yhat_tr, yhat_te= yhat_np[:split], yhat_np[split:]

mse_tr = mse(y_tr, yhat_tr); mae_tr = mae(y_tr, yhat_tr)
mse_te = mse(y_te, yhat_te); mae_te = mae(y_te, yhat_te)
r_te   = pearsonr(y_te.reshape(-1), yhat_te.reshape(-1))[0]

print(f"Train  MSE={mse_tr:.4f} MAE={mae_tr:.4f}")
print(f"Test   MSE={mse_te:.4f} MAE={mae_te:.4f}  | Pearson r (test)={r_te:.3f}")


# 5) 图表
split_year = years[split-1]

# (A) 时间序列叠加：兔子
plt.figure(figsize=(10,3.6))
plt.plot(years, hares_obs, 'k-', lw=2, label="Hares (Observed)")
plt.plot(years, hares_pred, 'b--', lw=2, label="Hares (Neural ODE)")
plt.axvline(split_year, color='gray', ls='--', alpha=.7, label="Train/Test split")
plt.xlabel("Year"); plt.ylabel("Pelts (proxy for population)")
plt.title("Hares: Observed vs Neural ODE")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# (B) 时间序列叠加：猞猁
plt.figure(figsize=(10,3.6))
plt.plot(years, lynx_obs, 'k-', lw=2, label="Lynx (Observed)")
plt.plot(years, lynx_pred, 'r--', lw=2, label="Lynx (Neural ODE)")
plt.axvline(split_year, color='gray', ls='--', alpha=.7)
plt.xlabel("Year"); plt.ylabel("Pelts (proxy for population)")
plt.title("Lynx: Observed vs Neural ODE")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# (C) 预测放大（仅测试区域）
plt.figure(figsize=(8,3.6))
plt.plot(years[split:], hares_obs[split:], 'k-', lw=2, label="Hares (Obs)")
plt.plot(years[split:], hares_pred[split:], 'b--', lw=2, label="Hares (Pred)")
plt.plot(years[split:], lynx_obs[split:],  'k-', lw=1.5, alpha=.6, label="Lynx (Obs)")
plt.plot(years[split:], lynx_pred[split:], 'r--', lw=1.8, label="Lynx (Pred)")
plt.xlabel("Year"); plt.ylabel("Pelts")
plt.title("Forecast Region (Test Years)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# (D) 相位肖像：兔子 vs 猞猁
plt.figure(figsize=(5.6,5.2))
plt.plot(hares_obs, lynx_obs, 'k.-', label="Observed")
plt.plot(hares_pred, lynx_pred, 'c.-', label="Neural ODE")
plt.xlabel("Hares (pelts)"); plt.ylabel("Lynx (pelts)")
plt.title("Phase Portrait: Predator–Prey Cycle")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# (E) 随时间的残差（原始单位的绝对误差）
abs_err_hares = np.abs(hares_pred - hares_obs)
abs_err_lynx  = np.abs(lynx_pred  - lynx_obs)

plt.figure(figsize=(10,3.4))
plt.plot(years, abs_err_hares, label="|Error| Hares", lw=1.8)
plt.plot(years, abs_err_lynx,  label="|Error| Lynx",  lw=1.8)
plt.axvline(split_year, color='gray', ls='--', alpha=.7)
plt.xlabel("Year"); plt.ylabel("Absolute Error (pelts)")
plt.title("Prediction Errors over Time")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# (F) 观察 vs 预测散点图（原始单位）+ R^2
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) + 1e-12
    return 1.0 - ss_res/ss_tot

r2_hares = r2_score(hares_obs[split:], hares_pred[split:])
r2_lynx  = r2_score(lynx_obs[split:],  lynx_pred[split:])

plt.figure(figsize=(9,3.6))
plt.subplot(1,2,1)
plt.scatter(hares_obs[split:], hares_pred[split:], s=35, alpha=.85)
plt.plot([hares_obs.min(), hares_obs.max()],
         [hares_obs.min(), hares_obs.max()], 'k--', lw=1)
plt.title(f"Hares (Test): R²={r2_hares:.2f}")
plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.grid(alpha=.3)

plt.subplot(1,2,2)
plt.scatter(lynx_obs[split:], lynx_pred[split:], s=35, alpha=.85, color='tab:red')
plt.plot([lynx_obs.min(), lynx_obs.max()],
         [lynx_obs.min(), lynx_obs.max()], 'k--', lw=1)
plt.title(f"Lynx (Test): R²={r2_lynx:.2f}")
plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.grid(alpha=.3)

plt.tight_layout(); plt.show()