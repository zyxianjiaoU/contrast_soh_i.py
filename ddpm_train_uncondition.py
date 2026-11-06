import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CosAMR
from torch.utils.data import DataLoader

import unet
from utils.diffusion import GaussianDiffusion
from utils.diffusion import get_beta_schedule
from utils.nn import LoadData


class Model:
    def __init__(self, betas: np.ndarray, dropout: float, batch, out_ch, local_bs):
        self.betas = betas
        self.diffusion = GaussianDiffusion(betas=betas)
        self.b = batch
        # model
        self.unet = unet.unet(num_res_blocks=1, ch=8, ch_mult=(1, 2, 4),
                        dropout=dropout, attn_resolutions=(2, 4, 8), batch=batch, out_ch=out_ch)
        for p in self.unet.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.local_bs = local_bs
        self.x_shape = [batch, 1, 64]


    def _denoise(self, x, t):
        n, c, l = x.shape
        assert x.dtype == torch.float32
        assert t.shape[0] == n and t.dtype in [torch.int32, torch.int64]
        ori_out_ch =  c
        out = self.unet(x, t)
        assert [i for i in out.shape] == [n, ori_out_ch, l]
        return out

    def train_fn(self,x):
        n, c, l = x.shape
        assert n == self.b
        t = torch.randint(0, self.diffusion.num_timestep, size=(n, ))
        losses = torch.mean(self.diffusion.p_losses(denoise_fn=self._denoise, x_start=x, t=t))
        return losses

    def samples_fn(self, dummy_noise):
        return {
            'samples': self.diffusion.p_sample_loop(
                denoise_fn=self._denoise,
                shape=list(dummy_noise.shape),
            )
        }

    def make_sampling(self):
        out = {}
        def _make_inputs(local_bs):
            # dummy inputs to feed to samplers
            input_x = torch.full([local_bs, *self.x_shape[1:]], fill_value=torch.nan)
            return input_x

        # samples
        samples_outputs = self.samples_fn(_make_inputs(self.local_bs))
        for k, v in samples_outputs.items():
            out['samples/{}'.format(k)] = v
        return out


bs = 16
l_rate = 5e-3
training_cycle = 2000
model = Model(dropout=0.1, batch=bs, out_ch=1, local_bs=bs,
              betas=get_beta_schedule(
                  beta_schedule='linear', beta_start=0.0001, beta_end=0.2,
                  num_diffusion_timesteps=1000)
              )
# optim = torch.optim.SGD(model.unet.parameters(), lr=l_rate, momentum=0.9)
optim = torch.optim.AdamW(model.unet.parameters(), lr=l_rate, weight_decay=1e-4)
lr_scheduler = CosAMR(optim, T_0=400, T_mult= 2, eta_min=1e-8)

data = torch.load('clip_data_100_1024.pt')
soh_train = data['soh_train']
data_train = LoadData(soh_train)
load_data_train = DataLoader(data_train, shuffle=True, batch_size=bs, drop_last=True)
# data_test = LoadData(capacity_test)
# load_data_test = DataLoader(data_test, shuffle=True, batch_size=bs, drop_last=True)

for epoch in range(training_cycle):
    total_loss = 0.00
    for data_pair in load_data_train:
        src = data_pair
        src = src.to(torch.float32)
        src[:, :, 0] = src[:, :, 1]
        result_loss = model.train_fn(src)
        total_loss += result_loss
        optim.zero_grad()
        result_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=500, norm_type=1)
        optim.step()
    lr_scheduler.step()
    if epoch % 10 == 0:
        print('epoch: {}'.format(epoch), total_loss.item())
torch.save(model.unet.state_dict(), 'checkpoints/ddpm.pkl')
model = Model(dropout=0.1, batch=bs, out_ch=1, local_bs=bs,
              betas=get_beta_schedule(
                  beta_schedule='linear', beta_start=0.0001, beta_end=0.2,
                  num_diffusion_timesteps=100)
              )
model.unet.load_state_dict(torch.load('checkpoints/ddpm.pkl'))
model.unet.eval()
sample = model.make_sampling()
samples = sample['samples/samples']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("model_ddpm_sample_1")
ax.set_xlabel("cycle")
ax.set_ylabel("SOH")
ax.scatter(range(len(samples[1][0])), samples[1, 0], c='green', s=1)
plt.show()