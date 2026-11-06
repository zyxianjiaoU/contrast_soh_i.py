"""物理约束的容量衰减曲线表示（physics-constrained curve representation）。

一个物理上合理的 SOH 曲线必须满足：
  1. SOH(0) = 1                      —— 新电池满容量
  2. SOH(t) 单调不增                  —— 容量只能衰减
  3. SOH(t) ∈ [SOH_EOL, 1]           —— 有界，不会变成 0 或负数
  4. 曲线光滑，但允许出现拐点(knee)     —— 一段缓降后加速跳水

本模块把「任意无约束潜变量 z ∈ R^{B×2×(L-1)}」映射到「必然满足 1~4 的曲线」：

    z[:, 0] : 每步衰减增量 w 的 logits（标准化后）
    z[:, 1] : 总衰减量 D 的 logit      （标准化后）

解码 decode(z):
    u   = z0 * u_std + u_mean                 # 标准化还原
    w   = softmax(smooth(u))                  # w_t > 0, Σw_t = 1   → 增量恒正
    D   = d_min + (d_max-d_min)*sigmoid(z1)   # 总衰减有界          → SOH 有下界
    SOH = [1, 1 - D * cumsum(w)]              # 累积和单调递增      → SOH 单调不增
                                              # SOH(0) 由定义恒为 1
`softmax(smooth(u))` 里的平滑算子同时起到了带限低通的作用，
所以生成曲线天然光滑，而 w 的峰值位置自由 → 允许出现拐点(knee)。

于是**无论网络输出什么**，采样得到的曲线都自动是"从 1 开始、单调下降、
有界、可含拐点"的物理合理曲线。扩散模型只需要在潜空间里学分布，
物理规律由解码器硬保证，而不是靠网络"学会"。

编码 encode(soh) 是 decode 的逆，用于把训练数据变成扩散目标。
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """对 [B, L] 沿时间轴做高斯平滑（replicate 边界，不改变首尾量级）。"""
    if sigma <= 0:
        return x
    k = int(4 * sigma) | 1
    pad = k // 2
    t = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    kern = torch.exp(-t ** 2 / (2 * sigma ** 2))
    kern = kern / kern.sum()
    return F.conv1d(F.pad(x.unsqueeze(1), (pad, pad), mode='replicate'),
                    kern.view(1, 1, -1)).squeeze(1)


def clean_curves(soh_raw: torch.Tensor, channel: int = 0, sigma: float = 8.0,
                 multiple: int = 8):
    """把 info_battery.get_soh_from_q 产出的原始数据整理成可训练的 SOH 曲线。

    原始 soh 里有两个必须先修掉的问题：
      * soh[:, :, 0] 存的不是 SOH，而是电池寿命(bat_life)！见 get_soh_from_q
        里 `soh[0][0][0] = bat_life`，所以第 0 个点可能是 148~2233 这种数。
        这里把它单独取出来作为协变量，并从曲线里去掉。
      * 采样/量化噪声会让曲线局部回升（原始数据里 8.6% 的步长是上升的），
        而且寿命末端之外是线性外推的合成数据。这里先平滑再强制单调。

    multiple: UNet 有 log2(multiple) 次下采样，潜变量长度（曲线长度-1）必须能整除它。

    返回:
        soh   : [N, L]  单调不增、从 1 开始、取值在 [0, 1] 的曲线
        life  : [N]     电池实际寿命（cycle），可作为条件变量
    """
    life = soh_raw[:, channel, 0].clone()
    x = soh_raw[:, channel, 1:].to(torch.float32).clamp(0.0, 1.0)

    fade = (1.0 - x).clamp(min=0.0)
    fade = gaussian_smooth(fade, sigma)      # 去量化噪声
    fade = torch.cummax(fade, dim=1).values  # 强制单调（等价于保序回归的简化版）
    fade = fade - fade[:, :1]                # 归零 → SOH(0) = 1
    fade = fade.clamp(min=0.0)
    # 尾部补齐成常数（不再衰减），同时让 潜变量长度 = L-1 能被 multiple 整除
    target = ((fade.shape[1] // multiple) + 1) * multiple + 1
    if fade.shape[1] < target:
        fade = F.pad(fade, (0, target - fade.shape[1]), mode='replicate')
    return (1.0 - fade).clamp(0.0, 1.0), life


def project_monotone(soh: torch.Tensor) -> torch.Tensor:
    """事后修补：把任意曲线投影成单调不增、有界的曲线（cummin 即 L∞ 投影）。

    用于「已经训练好的普通 DDPM」——不改模型，只在采样后投影。
    代价是曲线会保留原来的锯齿，只是不再回升。
    """
    soh = torch.cummin(soh, dim=1).values
    return soh.clamp(0.0, 1.0)


def curve_metrics(soh: torch.Tensor) -> dict:
    """评估一批曲线的物理合理性。"""
    soh = soh.detach().float()
    d = torch.diff(soh, dim=1)
    return {
        'monotone_frac': (d <= 1e-6).all(dim=1).float().mean().item(),
        'max_rise': d.max(dim=1).values.mean().item(),
        'mean_abs_curv': torch.diff(d, dim=1).abs().mean().item(),
        'start': soh[:, 0].mean().item(),
        'start_std': soh[:, 0].std().item(),
        'end': soh[:, -1].mean().item(),
        'min': soh.min().item(),
        'max': soh.max().item(),
    }


# --------------------------------------------------------------------------- #
# 单调曲线编解码器
# --------------------------------------------------------------------------- #
class MonotoneCurveCodec:
    """潜变量 <-> SOH 曲线的双向映射，物理约束由结构保证。

    维度约定：曲线 [B, L]，潜变量 [B, 2, L-1]。
    """

    def __init__(self, length: int, d_min: float = 0.02, d_max: float = 1.0,
                 eps_frac: float = 0.02, sigma: float = 1.0):
        self.length = length
        self.d_min = d_min
        self.d_max = d_max
        # 增量下界：把 w 的下限设为「均匀分布的 eps_frac 倍」，
        # 避免 log(w) 出现 -20 这种极端值而破坏扩散模型的数值条件。
        self.eps = eps_frac / length
        self.sigma = sigma
        # 标准化统计量（fit_stats 填充）
        self.u_mean, self.u_std = 0.0, 1.0
        self.d_mean, self.d_std = 0.0, 1.0

    # ---------------- 编码 ----------------
    def _encode_raw(self, soh: torch.Tensor):
        """返回未标准化的 (u, D)。u 对应 L-1 个相邻步的衰减增量。"""
        fade = (1.0 - soh).clamp(min=0.0)
        fade = fade - fade[:, :1]                      # F(0)=0
        d = torch.diff(fade, dim=1)                    # [B, L-1] 增量 >= 0
        D = d.sum(dim=1).clamp_min(1e-6)               # 总衰减
        w = d / D[:, None]
        w = gaussian_smooth(w, self.sigma)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        u = torch.log(w + self.eps)
        return u, D

    def fit_stats(self, soh: torch.Tensor):
        u, D = self._encode_raw(soh)
        self.u_mean, self.u_std = u.mean().item(), u.std().item()
        ratio = ((D - self.d_min) / (self.d_max - self.d_min)).clamp(1e-3, 1 - 1e-3)
        raw = torch.log(ratio / (1 - ratio))
        self.d_mean, self.d_std = raw.mean().item(), raw.std().item()
        return self

    def encode(self, soh: torch.Tensor) -> torch.Tensor:
        u, D = self._encode_raw(soh)
        z0 = (u - self.u_mean) / self.u_std
        ratio = ((D - self.d_min) / (self.d_max - self.d_min)).clamp(1e-3, 1 - 1e-3)
        raw = torch.log(ratio / (1 - ratio))
        z1 = ((raw - self.d_mean) / self.d_std)[:, None].expand(-1, u.shape[1])
        return torch.stack([z0, z1], dim=1)

    # ---------------- 解码 ----------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B,2,L-1] -> [B,L]，输出必然 SOH(0)=1、单调不增、∈[SOH_EOL,1]。"""
        u = z[:, 0] * self.u_std + self.u_mean
        raw = z[:, 1].mean(dim=1) * self.d_std + self.d_mean
        D = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(raw)   # [B]
        w = torch.softmax(gaussian_smooth(u, self.sigma), dim=1)          # 正、和为1
        fade = D[:, None] * torch.cumsum(w, dim=1)                        # [B, L-1]
        ones = torch.ones(fade.shape[0], 1, device=fade.device, dtype=fade.dtype)
        return torch.cat([ones, 1.0 - fade], dim=1)                       # [B, L]

    def latent_shape(self, batch: int):
        return (batch, 2, self.length - 1)

    def __repr__(self):
        return (f'MonotoneCurveCodec(L={self.length}, D∈[{self.d_min},{self.d_max}], '
                f'u~({self.u_mean:.2f},{self.u_std:.2f}))')


# --------------------------------------------------------------------------- #
# 采样器：DDIM（与训练时相同的 T，但只走 num_steps 步）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddim_sample(diffusion, denoise_fn, *, shape, num_steps=100, eta=0.0,
                clip_denoised=False, device=None):
    """确定性 DDIM 采样。

    注意：扩散参数（betas/alpha_cumprod）必须与训练时**完全一致**。
    原脚本训练用 1000 步、采样却新建了一个 100 步的 GaussianDiffusion，
    模型看到的时间步含义完全变了，这是生成结果面目全非的主因之一。

    这里复用训练时的 diffusion 对象，只把时间步稀疏化成 num_steps 步。
    """
    if device is None:
        try:
            device = next(denoise_fn.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device('cpu')
    # 时间步子序列：T-1, ..., 0，共 num_steps 个
    ts = torch.linspace(diffusion.num_timestep - 1, 0, num_steps).round().long().to(device)
    x = torch.randn(shape, device=device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)

    for i, t in enumerate(ts):
        t_batch = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
        eps = denoise_fn(x, t_batch)
        ab = alpha_cumprod[t_batch].view(-1, 1, 1)
        x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        if clip_denoised:
            x0 = x0.clamp(0.0, 1.0)
        if i == len(ts) - 1:
            x = x0
            break
        ab_next = alpha_cumprod[ts[i + 1]].view(1, 1, 1)
        # DDIM(eta=0)：只沿预测的 x0 方向走
        x = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps
    return x
