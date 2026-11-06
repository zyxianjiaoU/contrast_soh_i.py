import torch
from torch import nn as NN
from torch.nn import functional as F

from utils import nn


def non_linearity(x):
  return F.sigmoid(x) * x


def normalize(x, num_groups):
  return F.group_norm(x, num_groups)


class upsample(NN.Module):
    def __init__(self, c, with_conv=True):
        super(upsample, self).__init__()
        self.with_conv = with_conv
        self.c = c
        self.conv1d = NN.Conv1d(c, c, 3, bias=True, padding='same')

    def forward(self, x):
        n, c, l = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest-exact')
        assert [i for i in x.shape] == [n, c, 2*l]
        if self.with_conv:
          x = self.conv1d(x)
          assert [i for i in x.shape] == [n, c, 2*l]
        return x


class downsample(NN.Module):
    def __init__(self, c, with_conv=True):
        super(downsample, self).__init__()
        self.with_conv = with_conv
        self.conv1d = NN.Conv1d(c, c, 3, bias=True, stride=2, padding=1)
        self.avg_pool1d = NN.AvgPool1d(2, 2)


    def forward(self, x):
        n, c, l = x.shape
        if self.with_conv:
          x = self.conv1d(x)
        else:
          x = self.avg_pool1d(x, 2, 2)
        assert [i for i in x.shape] == [n, c, l // 2]
        return x


class resnet_block(NN.Module):
    def __init__(self, c_in, dropout, ori_ch, c_out=None, conv_shortcut=False):
        super(resnet_block, self).__init__()
        if c_out is None:
            c_out = c_in
        self.c_in = c_in
        self.c_out = c_out
        self.conv1d_1 = NN.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding='same')
        self.dense = NN.Linear(4*ori_ch, c_out)
        self.dropout = NN.Dropout(p=dropout)
        self.conv1d_2 = NN.Conv1d(c_out, c_out, kernel_size=3, stride=1, padding='same')
        self.conv_shortcut = conv_shortcut
        self.conv1d_3 = NN.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding='same')
        self.nin = nn.nin(c_in, c_out)

    def forward(self, x, temb):
        n, c, l = x.shape
        assert self.c_in == c
        h = x
        h = non_linearity(normalize(h, 4))
        h = self.conv1d_1(h)
        # add in timestep embedding
        h += self.dense(non_linearity(temb))[:, :, None]
        h = non_linearity(normalize(h, 4))
        h = self.dropout(h)
        h = self.conv1d_2(h)
        if c != self.c_out:
            if self.conv_shortcut:
                x = self.conv1d_3(x)
            else:
                x = self.nin(x)
        assert x.shape == h.shape
        # print('x={}, temb={}'.format(x.shape, temb.shape))
        return x + h


class attn_block(NN.Module):
    def __init__(self, c_in):
        super(attn_block, self).__init__()
        self.L_1 = nn.nin(c_in, c_in)
        self.L_2 = nn.nin(c_in, c_in)
        self.L_3 = nn.nin(c_in, c_in)
        self.L_4 = nn.nin(c_in, c_in)
        self.c_in = c_in

    def forward(self, x):
        n, c, l = x.shape
        assert c == self.c_in
        h = normalize(x, 4)
        q = self.L_1(h).unsqueeze(-1)
        k = self.L_2(h).unsqueeze(-1)
        v = self.L_3(h).unsqueeze(-1)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (int(c) ** (-0.5))
        w = torch.reshape(w, [n, l, 1, l * 1])
        w = torch.softmax(w, -1)
        w = torch.reshape(w, [n, l, 1, l, 1])

        h = torch.einsum('bhwHW,bcHW->bchw', w, v).squeeze(-1)
        h = self.L_4(h)

        assert h.shape == x.shape
        # print('attn_shape', x.shape)
        return x + h


class t_embedding(NN.Module):
    def __init__(self, c_in, batch):
        super(t_embedding, self).__init__()
        self.c_in = c_in
        self.c_out = 4 * c_in
        self.batch = batch
        self.dense1 = NN.Linear(c_in, self.c_out, bias=True)
        self.dense2 = NN.Linear(self.c_out, self.c_out, bias=True)

    def forward(self, t):
        # Timestep embedding
        t_emb = nn.get_timestep_embedding(t, self.c_in)
        # t_emb.shape == [t.shape[0], c_out]
        t_emb = self.dense1(t_emb)
        t_emb = self.dense2(non_linearity(t_emb))
        # 不要断言等于训练时的 batch：采样时的 batch 通常不同
        assert [i for i in t_emb.shape] == [t.shape[0], self.c_out]
        return t_emb


def make_layer(blk_in_channel, blk_mid_channel, dropout, ori_ch):
    block_list = [resnet_block(c_in=blk_in_channel, dropout=dropout, c_out=blk_mid_channel, ori_ch=ori_ch),
                  attn_block(c_in=blk_mid_channel)]
    return NN.Sequential(*block_list)


class sample_layer(NN.ModuleList):
    def __init__(self, ch_mult, num_resolutions, ch, mode: str,with_conv=True):
        super(sample_layer, self).__init__()
        assert mode in ['downsample', 'upsample']
        if mode == 'downsample':
            self.layer = NN.Sequential(
                *[downsample(c=ch_mult[i] * ch, with_conv=with_conv) for i in range(num_resolutions)])
        if mode == 'upsample':
            self.layer = NN.Sequential(
                *[upsample(c=ch_mult[i] * ch, with_conv=with_conv) for i in reversed(range(num_resolutions))])

    def forward(self, h, i_level):
        h = self.layer[i_level](h)
        return h


class down_sample_block(NN.Module):
    def __init__(self, num_res_blocks,ch, ch_mult, dropout, attn_resolutions, num_resolutions):
        super(down_sample_block, self).__init__()
        # Down sampling layer
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions
        self.layers = NN.Sequential(*[NN.Sequential() for _ in range(num_resolutions)])
        blk_in_channel = ch
        for j in range(num_resolutions):
            for idx in range(num_res_blocks):
                self.layers[j].add_module(
                    f"residule_layer{idx}",
                    make_layer(blk_in_channel=blk_in_channel,
                               blk_mid_channel=ch_mult[j]*ch,
                               dropout=dropout,
                               ori_ch=ch),
                )
                blk_in_channel = ch_mult[j]*ch
        self.ds_layer = sample_layer(ch_mult=ch_mult,
                                     num_resolutions=num_resolutions,
                                     ch=ch,
                                     with_conv=True,
                                     mode='downsample')

    def forward(self, hs, t):
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for block in self.layers[i_level]:
                h = block[0](hs[-1], t)
                # 进入attn，条件1：偶数次循环; 条件2：h符合要求
                if h.shape[-1] in self.attn_resolutions:
                    h = block[1](h) # attn_block
                # print('hs(ds).size: ', h.shape)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                hs.append(self.ds_layer(hs[-1], i_level))
                # print('hs(ds).size: ', hs[-1].shape)
        return hs


class up_sample_block(down_sample_block):
    def __init__(self, num_res_blocks, ch, ch_mult, dropout, attn_resolutions, num_resolutions):
        super().__init__(num_res_blocks,ch, ch_mult, dropout, attn_resolutions, num_resolutions)
        # # up sample layer
        ch_hs = sum([[i*ch, i*ch] for i in reversed(ch_mult)], [])[1:] + [ch]  # channel of hs
        c_mid = ch_mult[num_resolutions - 1] * ch
        for j in range(num_resolutions):
            for idx in range(num_res_blocks + 1):
                self.layers[j].add_module(
                    f"residule_layer{idx}",
                    make_layer(blk_in_channel= c_mid + ch_hs[j * (num_res_blocks + 1) + idx],   # channel of h+hs
                               blk_mid_channel= ch_mult[num_resolutions - 1 - j] * ch, # channel of h
                               dropout=dropout,
                               ori_ch=ch))
                c_mid = ch_mult[num_resolutions - 1 - j] * ch
        self.us_fn = sample_layer(ch_mult=ch_mult,
                                  num_resolutions=num_resolutions,
                                  ch=ch,
                                  mode='upsample')

    def forward(self, *args, hs, t):
        h = args[0]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for block in self.layers[i_level]:
                h = block[0](torch.concatenate((h, hs.pop()), dim=1), t)
                if h.shape[-1] in self.attn_resolutions: # 进入attn，条件1：偶数次循环  条件2：h符合要求
                    h = block[1](h) # attn_block
            # Up sample
            if i_level != (self.num_resolutions-1):
                h = self.us_fn(h, i_level)
            # if i_level == 3:
            #     print()
        return h


class full_connect(NN.Module):
    def __init__(self, c_in, c_out=1):
        super(full_connect, self).__init__()
        self.conv1d = NN.Conv1d(c_in, c_out, 3, bias=True, padding='same')

    def forward(self, h):
       h = non_linearity(normalize(h, 4))
       h = self.conv1d(h)
       return h


class unet(NN.Module):
    def __init__(self, num_res_blocks,ch, ch_mult, dropout, attn_resolutions, batch, out_ch=1,
                 in_ch=1):
        super(unet, self).__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        num_resolutions = len(ch_mult)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1d_init = NN.Conv1d(in_ch, ch, kernel_size=3, stride=1, padding='same')
        self.t_emb_fn = t_embedding(c_in=ch, batch=batch)
        self.ds = down_sample_block(num_res_blocks, ch, ch_mult, dropout, attn_resolutions, num_resolutions)
        c_mid = ch*ch_mult[-1]
        self.res_mid_1 = resnet_block(c_in=c_mid, dropout=dropout, ori_ch=ch)
        self.attn_mid = attn_block(c_mid)
        self.res_mid_2 = resnet_block(c_in=c_mid, dropout=dropout, ori_ch=ch)
        self.us = up_sample_block(num_res_blocks, ch, ch_mult, dropout, attn_resolutions, num_resolutions)
        self.conv1d_final = full_connect(c_in=ch*ch_mult[0], c_out=out_ch)


    def _down_sample(self, x, t_emb):
       hs = [self.conv1d_init(x)]   # ch: 1 -> ch
       # Residual blocks for this resolution
       hs = self.ds(hs, t_emb)
       return hs

    def _middle(self, hs, t_emb):
       h = hs[-1]
       h = self.res_mid_1(h, t_emb)
       h = self.attn_mid(h)
       h = self.res_mid_2(h, t_emb)
       return h

    def _up_sample(self, h, hs, t_emb):
        assert h.shape[1] == self.ch*self.ch_mult[-1]
        h = self.us(h, hs=hs, t=t_emb)
        return h

    def forward(self, x, t):
        assert x.dtype == torch.float32
        assert t.dtype in [torch.int32, torch.int64]
        t_emb = self.t_emb_fn(t)
        hs = self._down_sample(x, t_emb)
        h = self._middle(hs, t_emb)
        h = self._up_sample(h, hs, t_emb)
        h = self.conv1d_final(h)
        assert [i for i in h.shape] == [x.shape[0], self.out_ch, x.shape[-1]]
        return h

