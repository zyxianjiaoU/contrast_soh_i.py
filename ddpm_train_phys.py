"""无条件扩散模型生成容量衰减曲线 —— 物理约束版训练脚本。

与 ddpm_train_uncondition.py 的区别（每一条都对应一个真实 bug 或建模缺陷）：

  1. 数据修复   : soh[:, :, 0] 存的是电池寿命而不是 SOH，原脚本用
                  `src[:, :, 0] = src[:, :, 1]` 硬覆盖，而且通道数是 2，
                  与 Conv1d(1, ...) 不匹配。这里用 physics.clean_curves 正规处理。
  2. 序列长度   : 训练用 512 点，采样却用 x_shape=[B, 1, 64]，长度对不上。
  3. 时间步一致 : 训练用 1000 步的 betas，采样又新建了一个 100 步的
                  GaussianDiffusion，模型看到的 t 含义完全变了 —— 结果必然是噪声。
  4. 物理约束   : 不直接生成 SOH，而是生成「衰减增量」的潜变量，由解码器
                  保证 SOH(0)=1、单调不增、有界、可含拐点（见 physics.py）。
  5. 全局感受野 : attn_resolutions=(2,4,8) 与 L=512 完全不匹配，注意力层从未生效；
                  改为在最粗分辨率上开注意力，让模型能看到整条曲线的走向。

用法:
    python ddpm_train_phys.py --model both --steps 6000
"""

import argparse
import os
import time

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import unet
from physics import (MonotoneCurveCodec, clean_curves, curve_metrics,
                     ddim_sample, project_monotone)
from utils.diffusion import GaussianDiffusion, get_beta_schedule
from utils.nn import LoadData

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, 'clip_data_100_1024.pt')


# --------------------------------------------------------------------------- #
def build_unet(length, in_ch, out_ch, batch, ch, ch_mult, num_res_blocks, dropout):
    """按实际序列长度推算注意力应放在哪一层（原脚本写死 (2,4,8)，永不触发）。"""
    resolutions = [length // (2 ** i) for i in range(len(ch_mult))]
    attn_resolutions = (min(resolutions[-1], 64),)
    net = unet.unet(num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                    dropout=dropout, attn_resolutions=attn_resolutions,
                    batch=batch, out_ch=out_ch, in_ch=in_ch)
    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return net, attn_resolutions


class EMA:
    """参数的指数滑动平均，显著改善扩散模型采样质量。"""

    def __init__(self, module, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module):
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, module):
        module.load_state_dict(self.shadow)


def train_ddpm(x_train, net, diffusion, steps, batch, lr, device, log_every=500,
               tag='model'):
    loader = DataLoader(LoadData(x_train), batch_size=batch, shuffle=True,
                        drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.05)
    ema = EMA(net.to(device))
    net.train()
    it = iter(loader)
    t0 = time.time()
    for step in range(steps):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader)
            x = next(it)
        x = x.to(device)
        t = torch.randint(0, diffusion.num_timestep, (x.shape[0],), device=device)
        loss = diffusion.p_losses(denoise_fn=net, x_start=x, t=t).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        ema.update(net)
        if (step + 1) % log_every == 0 or step == 0:
            print(f'[{tag}] step {step + 1:5d}/{steps}  loss={loss.item():.4f}  '
                  f'lr={sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s',
                  flush=True)
    net.eval()
    return ema


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['phys', 'plain', 'both'], default='both')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ch', type=int, default=16)
    ap.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 2, 4])
    ap.add_argument('--num_res_blocks', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--sample_steps', type=int, default=100)
    ap.add_argument('--n_samples', type=int, default=24)
    ap.add_argument('--out', default=os.path.join(HERE, 'samples_phys_comparison.png'))
    ap.add_argument('--save', default=os.path.join(HERE, 'checkpoints', 'ddpm_phys.pt'))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.steps = min(args.steps, 40)
        args.n_samples = min(args.n_samples, 8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    print('device:', device)

    # ---------------- 数据 ----------------
    raw = torch.load(DATA_PATH, weights_only=True)['soh_train'].to(torch.float32)
    soh, life = clean_curves(raw)
    length = soh.shape[1]
    print(f'curves {tuple(soh.shape)}  life {life.min():.0f}..{life.max():.0f}')
    print('cleaned data metrics:', curve_metrics(soh))

    betas = get_beta_schedule('linear', beta_start=1e-4, beta_end=0.02,
                              num_diffusion_timesteps=args.T)
    diffusion = GaussianDiffusion(betas=betas).to(device)

    results, nets = {}, {}

    # ---------------- 物理约束模型：在潜空间扩散 ----------------
    if args.model in ('phys', 'both'):
        codec = MonotoneCurveCodec(length).fit_stats(soh)
        print(codec)
        z_train = codec.encode(soh)
        # 编码-解码往返误差（应该很小，说明没有信息损失）
        rec_err = (codec.decode(z_train) - soh).abs().max().item()
        print(f'codec round-trip max error: {rec_err:.4f}')

        net, attn = build_unet(length, in_ch=2, out_ch=2, batch=args.batch,
                               ch=args.ch, ch_mult=args.ch_mult,
                               num_res_blocks=args.num_res_blocks,
                               dropout=args.dropout)
        print('attn_resolutions:', attn, ' params:',
              sum(p.numel() for p in net.parameters()))
        ema = train_ddpm(z_train, net, diffusion, args.steps, args.batch,
                         args.lr, device, tag='phys')
        ema.copy_to(net)
        if not args.no_save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save({'unet': net.state_dict(), 'codec': codec.__dict__}, args.save)

        z = ddim_sample(diffusion, net, shape=codec.latent_shape(args.n_samples),
                        num_steps=args.sample_steps, clip_denoised=False)
        samples = codec.decode(z)
        results['physics-constrained'] = samples
        nets['phys'] = net

    # ---------------- 对照：直接在 SOH 上扩散（修好 bug、但不加约束） --------
    if args.model in ('plain', 'both'):
        # UNet 要求序列长度能被 2^(层数-1) 整除，512=2^9；物理模型是 513（潜变量 512）
        x_plain = soh[:, :-1].unsqueeze(1)
        net2, attn = build_unet(x_plain.shape[-1], in_ch=1, out_ch=1, batch=args.batch,
                                ch=args.ch, ch_mult=args.ch_mult,
                                num_res_blocks=args.num_res_blocks,
                                dropout=args.dropout)
        print('plain attn_resolutions:', attn, ' params:',
              sum(p.numel() for p in net2.parameters()))
        ema2 = train_ddpm(x_plain, net2, diffusion, args.steps, args.batch,
                          args.lr, device, tag='plain')
        ema2.copy_to(net2)
        x = ddim_sample(diffusion, net2, shape=(args.n_samples, 1, x_plain.shape[-1]),
                        num_steps=args.sample_steps, clip_denoised=True)
        raw_samples = x[:, 0]
        results['plain DDPM'] = raw_samples
        results['plain DDPM + projection'] = project_monotone(raw_samples)
        nets['plain'] = net2

    # ---------------- 报告 ----------------
    print('\n=== physical-plausibility metrics ===')
    print(f"{'model':28s} {'monotone':>8s} {'max-rise':>9s} {'curvature':>10s} "
          f"{'start':>7s} {'end':>7s} {'min':>7s}")
    print(f"{'training data':28s} {curve_metrics(soh)['monotone_frac']:8.2f} "
          f"{curve_metrics(soh)['max_rise']:9.4f} {curve_metrics(soh)['mean_abs_curv']:10.2e} "
          f"{curve_metrics(soh)['start']:7.3f} {curve_metrics(soh)['end']:7.3f} "
          f"{curve_metrics(soh)['min']:7.3f}")
    for name, s in results.items():
        m = curve_metrics(s)
        print(f"{name:28s} {m['monotone_frac']:8.2f} {m['max_rise']:9.4f} "
              f"{m['mean_abs_curv']:10.2e} {m['start']:7.3f} {m['end']:7.3f} "
              f"{m['min']:7.3f}")

    # ---------------- 画图 ----------------
    if args.no_save:
        return
    order = [k for k in ['plain DDPM', 'plain DDPM + projection',
                         'physics-constrained'] if k in results]
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(4.0 * (len(order) + 1), 3.6),
                             sharey=True)
    ax = axes[0]
    for i in range(min(40, soh.shape[0])):
        ax.plot(soh[i].numpy(), color='0.6', lw=0.7, alpha=0.7)
    ax.set_title(f'training data (n={soh.shape[0]})')
    ax.axhline(0.8, color='k', ls='--', lw=0.7)
    for ax, name in zip(axes[1:], order):
        for i in range(results[name].shape[0]):
            ax.plot(results[name][i].detach().cpu().numpy(), lw=0.9, alpha=0.85)
        ax.set_title(name)
        ax.axhline(0.8, color='k', ls='--', lw=0.7)
    for ax in axes:
        ax.set_xlabel('cycle index (step 2)')
    axes[0].set_ylabel('SOH')
    fig.suptitle('DDPM capacity-fade sampling: effect of physics constraints')
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print('\nfigure saved:', args.out)

    np.save(os.path.splitext(args.out)[0] + '.npy',
            {k: v.detach().cpu().numpy() for k, v in results.items()}, allow_pickle=True)


if __name__ == '__main__':
    main()
