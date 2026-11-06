import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Conv1d, Conv3d, Flatten, Linear, LeakyReLU, ReLU, BatchNorm1d, MaxPool1d, BatchNorm3d, \
    AdaptiveAvgPool3d, BatchNorm2d

from tf_model import MultiheadAttention, LayerNorm, subsequent_mask
from utils.nn import PositionalEncoding, transform


class CapacityPre(nn.Module):
    def __init__(self):
        super(CapacityPre, self).__init__()
        self.model = Sequential(
            Conv1d(1, 32, kernel_size=5, stride=1),
            LeakyReLU(inplace=True, negative_slope=0.01),
            Conv1d(32, 64, kernel_size=5, stride=1),
            LeakyReLU(inplace=True, negative_slope=0.01),
            Conv1d(64, 16, kernel_size=5, stride=1),
            LeakyReLU(inplace=True, negative_slope=0.01),
            Flatten(),
            Linear(3904, 512),
            LeakyReLU(inplace=True, negative_slope=0.01),
            Linear(512, 384),
            LeakyReLU(inplace=True, negative_slope=0.01),
            Linear(384, 256),
        )

    def forward(self, x1):
        x1 = self.model(x1)
        return x1


class mode_embedding(nn.Module):
    def __init__(self, in_channel, out_dim):
        super(mode_embedding, self).__init__()
        # 定义卷积层
        self.features = nn.Sequential(
            Conv3d(in_channel, 4*in_channel, kernel_size=3, padding='same'),
            ReLU(inplace=True),
            BatchNorm3d(4*in_channel),
            Conv3d(4*in_channel, in_channel, kernel_size=3, padding='same'),
            ReLU(inplace=True),
            BatchNorm3d(in_channel),
            AdaptiveAvgPool3d((100, 8, 8)),
        )
        # 全连接层
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.05),
            Linear(64, out_dim),
            ReLU(inplace=True),
            BatchNorm2d(in_channel),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=3)
        x = self.classifier(x)
        return x


class SohEmbedding(nn.Module):
    def __init__(self, in_channel, out_dim):
        super(SohEmbedding, self).__init__()
        # 定义卷积层和池化层
        self.features = nn.Sequential(
            Conv1d(in_channel, 64, kernel_size=5, stride=4, padding=2),
            ReLU(inplace=True),
            BatchNorm1d(64),
            MaxPool1d(kernel_size=3, stride=2),
            Conv1d(64, 192, kernel_size=5, stride=1, padding=2),
            ReLU(inplace=True),
            BatchNorm1d(192),
            MaxPool1d(kernel_size=3, stride=2),
            Conv1d(192, 384, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm1d(384),
            Conv1d(384, 256, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm1d(256),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm1d(256),
            MaxPool1d(kernel_size=3, stride=2),
        )
        # 自适应层，将上一层的数据转换成6x6大小
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        # self.avg_pool = nn.AdaptiveAvgPool1d(6)
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            Linear(6 * 6, 256),
            ReLU(inplace=True),
            BatchNorm1d(256),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = MultiheadAttention(head=n_head, embedding_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, 4 * d_model)),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(4 * d_model, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x, attn_mask):
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, mask=attn_mask)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CnnTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, channel:int, training_mode:str):
        super(CnnTransformer, self).__init__()
        self.channel = channel
        self.training_mode = training_mode
        self.embedding = nn.Sequential(
            Linear(1, width),
            QuickGELU(),
            Linear(width, width),
            QuickGELU(),
        )
        # CNN 下采样：512 -> 64
        self.conv = nn.Sequential(
            nn.Conv1d(channel, 64, kernel_size=8, stride=4, padding=2),  # 512 -> 128
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.GELU(),
            nn.Conv1d(128, width, kernel_size=3, stride=1, padding=1),  # 64 -> 64
        )
        self.position = PositionalEncoding(max_len=5000, d_model=width, dropout=0.1)
        self.res_block = nn.ModuleList([
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])
        self.ln_final = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x.unsqueeze(-1))
        x = self.position(x)
        x = self.ln_final(x)
        x_new = torch.zeros((x.shape[0], self.channel * x.shape[2], x.shape[3])).to(device=x.device)
        x_new[:, 0::2] = x[:, 0]
        x_new[:, 1::2] = x[:, 1]
        attn_mask = subsequent_mask(x_new.shape[1])
        for block in self.res_block:
            x_new = block(x_new, attn_mask)
        return x_new

class SohEncoder(nn.Module):
    def __init__(self,embed_dim, d_model, channel:int, training_mode:str):
        super(SohEncoder, self).__init__()
        # self.encode_soh = SohEmbedding(out_dim=embed_dim)
        self.encode_soh = CnnTransformer(width=d_model, layers=1, heads=4, channel=channel, training_mode=training_mode)
        self.projection_charge = nn.Parameter(torch.randn((d_model, embed_dim), requires_grad=True))
        self.projection_discharge = nn.Parameter(torch.randn((d_model, embed_dim), requires_grad=True))
        self.projection = nn.Parameter(torch.randn((d_model, embed_dim), requires_grad=True))


    def forward(self, soh):
         # 2 channel
        x = self.encode_soh(soh)
        x = x[torch.arange(x.shape[0]), soh.argmax(dim=-1)[:, 0]] @ self.projection
        return x

class CurrentFeature(nn.Module):
    def __init__(self, embed_dim=512, d_model=512, memory_size=1024, momentum=0.999, training_mode="contrastive"):
        super(CurrentFeature, self).__init__()
        self.training_mode = training_mode
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.encode_current = ResNet18_2d(out_dim=embed_dim)
        self.soh_encoder = SohEncoder(embed_dim=embed_dim, d_model=d_model, channel=2, training_mode=self.training_mode)
        # 动量编码器（不可直接训练，EMA更新）
        self.current_encoder_m = copy.deepcopy(self.encode_current)
        self.soh_encoder_m = copy.deepcopy(self.soh_encoder)
        for param in self.current_encoder_m.parameters():
            param.requires_grad = False
        for param in self.soh_encoder_m.parameters():
            param.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(np.log([1 / 0.07])))
        # Memory Bank 初始化
        self.memory_size = memory_size
        self.momentum = momentum
        # 初始化空队列（FIFO）
        self.register_buffer("current_queue", torch.randn(memory_size, embed_dim))
        self.register_buffer("soh_queue", torch.randn(memory_size, embed_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.current_queue = F.normalize(self.current_queue, dim=1)
        self.soh_queue = F.normalize(self.soh_queue, dim=1)
        self.register_buffer("queue_full", torch.zeros(1, dtype=torch.bool))


    @torch.no_grad()
    def _momentum_update(self):
        """动量更新编码器（EMA）"""
        for param_q, param_k in zip(self.encode_current.parameters(),
                                    self.current_encoder_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.soh_encoder.parameters(),
                                    self.soh_encoder_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, current_feats, soh_feats):
        """更新Memory Bank（FIFO队列）"""
        batch_size = current_feats.shape[0]
        # 获取当前指针位置
        ptr = int(self.queue_ptr)
        # 替换队列中的旧特征（循环写入）
        if ptr + batch_size <= self.memory_size:
            self.current_queue[ptr:ptr + batch_size] = current_feats
            self.soh_queue[ptr:ptr + batch_size] = soh_feats
        else:
            # 处理边界情况（batch跨到队列头部）
            first_part = self.memory_size - ptr
            self.current_queue[ptr:] = current_feats[:first_part]
            self.soh_queue[ptr:] = soh_feats[:first_part]
            remaining = batch_size - first_part
            self.current_queue[:remaining] = current_feats[first_part:]
            self.soh_queue[:remaining] = soh_feats[first_part:]
        # 更新指针
        ptr = (ptr + batch_size) % self.memory_size
        self.queue_ptr[0] = ptr
        # 标记队列已满
        if ptr == 0 or self.queue_full:
            self.queue_full[0] = True

    @torch.no_grad()
    def update_memory_banks(self, current, soh, mode):
        """在 optimizer.step() 之后调用"""
        # 动量编码器提取当前batch特征（无梯度，用于更新队列）
        self._momentum_update()
        with torch.no_grad():
            i_e_m = self.current_encoder_m(current, mode)
            soh_e_m = self.soh_encoder_m(soh)
            i_e_m = F.normalize(i_e_m, dim=1)
            soh_e_m = F.normalize(soh_e_m, dim=1)
        # 更新Memory Bank（将当前batch的特征入队）
        if self.training:
            self._dequeue_and_enqueue(i_e_m, soh_e_m)

    def contrastive_forward(self, current, soh, mode):
        i_e = self.encode_current(current, mode)
        soh_e = self.soh_encoder(soh)
        # soh_e = self.encode_soh(soh)
        # joint multimodal embedding  [n, d_e] && normalize
        i_e = i_e / i_e.norm(dim=1, keepdim=True)
        soh_e = soh_e / soh_e.norm(dim=1, keepdim=True)
        # scaled pairwise cosine similarities [n, n]
        logit_scale = self.logit_scale.exp().float( )
        # 当前电流特征 vs 所有历史SOH特征 [batch, memory_size]
        logits_current = logit_scale * (i_e @ self.soh_queue.t())
        # 当前SOH特征 vs 所有历史电流特征 [batch, memory_size]
        logits_soh = logit_scale * (soh_e @ self.current_queue.t())
        # 5. 生成标签：正样本是当前batch对应的样本（在队列中的位置）
        # 注意：这里假设当前batch的样本在队列中的位置是 ptr - batch_size 到 ptr
        batch_size = current.shape[0]
        ptr = int(self.queue_ptr)
        # 计算正样本在队列中的索引（循环队列）
        if ptr >= batch_size:
            labels = torch.arange(ptr - batch_size, ptr, device=current.device)
        else:
            # 处理环绕情况
            labels = torch.cat([
                torch.arange(self.memory_size - (batch_size - ptr), self.memory_size, device=current.device),
                torch.arange(0, ptr, device=current.device)
            ])
        return logits_soh, logits_current, labels
        # return logit_per_soh, logit_per_i

class SupervisedFeature(CurrentFeature):
    def __init__(self, embed_dim,training_mode):
        super(SupervisedFeature, self).__init__(embed_dim,training_mode=training_mode)
        # 分类头（用于监督学习）
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    def supervised_forward(self, current, soh, mode):
        i_e = self.encode_current(current, mode)
        prediction = self.regressor(i_e)
        bat_life = soh[:, 0, 0].unsqueeze(-1)
        return prediction, bat_life

class MixFeature(SupervisedFeature):
    def __init__(self, embed_dim:int=512, training_mode:str='contrastive'):
        super(MixFeature, self).__init__(embed_dim, training_mode)
        # 分类头（用于监督学习）
        self.training_mode = training_mode

    def forward(self, current, soh, mode):
        if self.training_mode == 'contrastive':
            contrastive_out = self.contrastive_forward(current, soh, mode)
            return {'contrastive': contrastive_out}
        elif self.training_mode == 'supervised':
            supervised_out = self.supervised_forward(current, soh, mode)
            return {'supervised': supervised_out}
        elif self.training_mode == 'hybrid':
            contrastive_out = self.contrastive_forward(current, soh, mode)
            supervised_out = self.supervised_forward(current, soh, mode)
            return {
                'contrastive': contrastive_out,
                'supervised': supervised_out
            }
        return None


def memory_bank_contrastive_loss(logits_current, logits_soh, labels,
                                 current_weight=0.5, soh_weight=0.5):
    """
    计算InfoNCE loss，正样本是labels指定的位置
    """
    loss_current = F.cross_entropy(logits_current, labels)
    loss_soh = F.cross_entropy(logits_soh, labels)

    return current_weight * loss_current + soh_weight * loss_soh

class ResNet_2d(nn.Module):
    def __init__(self, block, num_block, out_dim):
        super(ResNet_2d, self).__init__()
        self.residual_layers = 4  # 每个 "residual layer" 含多个 blocks，对应上面列表中的一行 (即 conv2_x, conv3_x, conv4_x 或 conv5_x)
        self.blk1_in_channels = 32  # 按照上面的列表，此处应填 64，但由于大网络训练起来耗时太长，此处我们酌情把全部通道都除以 2
        self.blk_mid_channels = [32, 64, 128, 256]  # 原先的通道数：[64, 128, 256, 512]
        self.blk_channels = [self.blk1_in_channels] + self.blk_mid_channels  # [32, 32, 64, 128, 256]
        self.blk_stride = [2, 2, 2, 2]  # 每个 residual layer 的 stride
        self.blk_channel_expansion = block.channel_expansion
        self.mode_encode = mode_embedding(in_channel=2, out_dim=100)
        self.transform = transform(c=2, size=(100, 100))
        # 第一个卷积层（独立于 residual layers 之外）
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.blk_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.blk1_in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # residual layers (打包在 self.layers 中)
        self.layers = nn.Sequential()
        for i in range(self.residual_layers):
            blk_in_channels = self.blk_channels[i] if i == 0 else self.blk_channels[i] * block.channel_expansion
            blk_mid_channels = self.blk_channels[i + 1]
            self.layers.add_module(f"residule_layer{i}",
                                   self._make_layer(block=block,  # block 种类：BasicBlock 或 BottleneckBlock
                                                    blk_in_channels=blk_in_channels,
                                                    blk_mid_channels=blk_mid_channels,
                                                    num_blocks=num_block[i],  # 该 residual layer 有多少个 blocks
                                                    stride=self.blk_stride[i])
                                   )
        # 最后的全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = nn.Linear(in_features=256*block.channel_expansion, out_features=out_dim)
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim)

    def _make_layer(self, block, blk_in_channels, blk_mid_channels, num_blocks, stride):
        block_list = []
        stride_list = [stride] + [1] * (num_blocks - 1)  # 每个 block 的 stride

        for block_idx in range(num_blocks):
            if block_idx != 0:  # 对于 residual layer 中非第一个 block: 调整其 blk_in_channels
                blk_in_channels = blk_mid_channels * block.channel_expansion
            block_list.append(
                block(blk_in_channels=blk_in_channels,
                      blk_mid_channels=blk_mid_channels,
                      stride=stride_list[block_idx])
            )

        return nn.Sequential(*block_list)  # 返回一个 residual layer

    def forward(self, t, mode):
        # conv1
        out = torch.concatenate((self.transform(t), self.mode_encode(mode)), dim=1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        # "residual layers"（打包在 self.layers 中）
        out = self.layers(out)
        # average pooling
        out = self.avg_pool(out)  # shape of "out" before pooling (ResNet18): (batch_size, 1024, 4, 4)
        # linear layer
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out

class BasicBlock_2d(nn.Module):
    channel_expansion = 1  # {扩展后的最终输出通道数} / {扩展前的输出通道数（blk_mid_channels）}

    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(BasicBlock_2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=blk_in_channels,  # blk_in_channels：block 中第一个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # blk_mid_channels：block 中第一个 conv 层的输出通道数
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               bias=True)  # stride 可以任意指定
        self.bn1 = nn.BatchNorm2d(blk_mid_channels)

        self.conv2 = nn.Conv2d(in_channels=blk_mid_channels,  # block 中第二个 conv 层的输入通道数
                               out_channels=blk_mid_channels * self.channel_expansion,  # 扩展后的最终输出通道数
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)  # stride 恒为 1
        self.bn2 = nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or blk_in_channels != self.channel_expansion * blk_mid_channels:  # 形状不同
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=blk_in_channels,
                          out_channels=self.channel_expansion * blk_mid_channels,  # 变换通道数
                          kernel_size=1,
                          padding=0,
                          stride=stride,
                          bias=True),  # 变换空间维度
                nn.BatchNorm2d(self.channel_expansion * blk_mid_channels)
            )
        else:  # 形状相同
            self.shortcut = nn.Sequential()

    def forward(self, t):

        # conv1
        out = self.conv1(t)
        out = self.bn1(out)
        out = self.relu(out)

        # conv2 & shortcut
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(t)
        out = self.relu(out)
        return out


class BottleneckBlock_2d(nn.Module):
    channel_expansion = 4  # {扩展后的最终输出通道数} / {扩展前的输出通道数（blk_mid_channels）}

    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(BottleneckBlock_2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=blk_in_channels,  # blk_in_channels：block 中第一个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # blk_mid_channels：block 中第一个 conv 层的输出通道数
                               kernel_size=1,
                               bias=False)  # stride 恒为 1
        self.bn1 = nn.BatchNorm2d(blk_mid_channels)

        self.conv2 = nn.Conv2d(in_channels=blk_mid_channels,  # block 中第二个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # block 中第二个 conv 层的输出通道数
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               bias=False)  # stride 可以任意指定
        self.bn2 = nn.BatchNorm2d(blk_mid_channels)

        self.conv3 = nn.Conv2d(in_channels=blk_mid_channels,  # block 中第三个 conv 层的输入通道数
                               out_channels=blk_mid_channels * self.channel_expansion,  # 扩展后的最终输出通道数
                               kernel_size=1,
                               bias=False)  # stride 恒为 1
        self.bn3 = nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)
        self.relu = nn.ReLU(inplace=True)

        # 实现 shortcut connection
        if stride != 1 or blk_in_channels != blk_mid_channels * self.channel_expansion:  # 形状不同
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=blk_in_channels,
                          out_channels=blk_mid_channels * self.channel_expansion,  # 变换空间维度
                          kernel_size=1,
                          padding=0,
                          stride=stride),  # 变换空间维度
                nn.BatchNorm2d(blk_mid_channels * self.channel_expansion)
            )
        else:  # 形状相同
            self.shortcut = nn.Sequential()

    def forward(self, t):
        # conv1
        out = self.conv1(t)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # conv3 & shortcut
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(t)
        out = self.relu(out)
        return out


class BottleneckBlock_3d(nn.Module):
    channel_expansion = 4  # {扩展后的最终输出通道数} / {扩展前的输出通道数（blk_mid_channels）}

    def __init__(self, blk_in_channels, blk_mid_channels, stride=1):
        super(BottleneckBlock_3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=blk_in_channels,  # blk_in_channels：block 中第一个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # blk_mid_channels：block 中第一个 conv 层的输出通道数
                               kernel_size=1,
                               bias=False)  # stride 恒为 1
        self.bn1 = nn.BatchNorm3d(blk_mid_channels)

        self.conv2 = nn.Conv3d(in_channels=blk_mid_channels,  # block 中第二个 conv 层的输入通道数
                               out_channels=blk_mid_channels,  # block 中第二个 conv 层的输出通道数
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               bias=False)  # stride 可以任意指定
        self.bn2 = nn.BatchNorm3d(blk_mid_channels)

        self.conv3 = nn.Conv3d(in_channels=blk_mid_channels,  # block 中第三个 conv 层的输入通道数
                               out_channels=blk_mid_channels * self.channel_expansion,  # 扩展后的最终输出通道数
                               kernel_size=1,
                               bias=False)  # stride 恒为 1
        self.bn3 = nn.BatchNorm3d(blk_mid_channels * self.channel_expansion)
        self.relu = nn.ReLU(inplace=True)

        # 实现 shortcut connection
        if stride != 1 or blk_in_channels != blk_mid_channels * self.channel_expansion:  # 形状不同
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels=blk_in_channels,
                          out_channels=blk_mid_channels * self.channel_expansion,  # 变换空间维度
                          kernel_size=1,
                          padding=0,
                          stride=stride),  # 变换空间维度
                nn.BatchNorm3d(blk_mid_channels * self.channel_expansion)
            )
        else:  # 形状相同
            self.shortcut = nn.Sequential()

    def forward(self, t):
        # conv1
        out = self.conv1(t)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # conv3 & shortcut
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(t)
        out = self.relu(out)
        return out


def ResNet18_2d(out_dim):
    return ResNet_2d(block=BottleneckBlock_2d, num_block=[2, 2, 2, 2], out_dim=out_dim)


















