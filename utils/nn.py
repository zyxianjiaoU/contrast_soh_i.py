import math
import string

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import dataset


# ===== Utilities =====
class LoadData(dataset.Dataset):

    def __init__(self, q):
        super(LoadData, self).__init__()
        self.capacity = q

    def __getitem__(self, index):
        label = self.capacity[index]
        return label

    def __len__(self):
        return len(self.capacity)

class LoadPairs(dataset.Dataset):
    def __init__(self, f, q, m):
        super(LoadPairs, self).__init__()
        self.feature = f
        self.capacity = q
        self.mode = m

    def __getitem__(self, index):
        input = self.feature[index]
        label = self.capacity[index]
        mode = self.mode[index]
        return input, label, mode

    def __len__(self):
        return len(self.feature)


def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = torch.exp(torch.arange(0, half_dim, device=timesteps.device) * -emb)
  emb = timesteps[:, None] * emb[None, :]
  emb = torch.concatenate((torch.sin(emb), torch.cos(emb)), dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = torch.concatenate((emb, torch.tensor([[0, 0], [0, 1]],
                                               device=timesteps.device)), dim=1)
  assert [i for i in emb.shape] == [timesteps.shape[0], embedding_dim]
  return emb

class nin(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(nin, self).__init__()
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.rand(in_dim, self.out_dim))
        self.b = nn.Parameter(torch.rand(self.out_dim, 1))

    def contract_inner(self, x, y):
      """tensordot(x, y, 1)."""
      x_chars = list(string.ascii_lowercase[:len(x.shape)])
      y_chars = list(string.ascii_uppercase[:len(y.shape)])
      assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
      y_chars[0] = x_chars[-2]
      out_chars = x_chars[:-2] + [y_chars[-1]] + [x_chars[-1]]
      return self._einsum(x_chars, y_chars, out_chars, x, y)


    def _einsum(self, a, b, c, x, y):
      einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
      return torch.einsum(einsum_str, x, y)

    def forward(self, x):
        y = self.contract_inner(x, self.W) + self.b
        assert [i for i in y.shape] == [x.shape[0], self.out_dim, x.shape[-1]]
        return y

def meanflat(x):
  return torch.mean(x, dim=list(range(1, len(x.shape))))

class transform(nn.Module):
    def __init__(self, c, size, with_conv=True):
        super(transform, self).__init__()
        self.with_conv = with_conv
        self.size = size
        self.conv2d = nn.Conv2d(c, c, kernel_size=3, bias=True, padding='same')

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, size=self.size, mode='bicubic')
        if self.with_conv:
          x = self.conv2d(x)
        assert [i for i in x.shape] == [n, c, self.size[0], self.size[0]]
        return x


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
        math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, max_len, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x[:, :] = x[:, :] + Variable(self.pe[:, :x.size()[2]], requires_grad=False)
        return self.dropout(x)


def linear_evaluation(model, train_loader, test_loader):
    """冻结 encoder，训练线性分类器"""
    std = train_loader.dataset.std_life
    mean = train_loader.dataset.mean_life
    model.eval()
    # 提取特征
    def extract_features(loader, state):
        features, labels = torch.empty((0, model.embed_dim)), torch.empty((0, 1))
        with torch.no_grad():
            if state == "train":
                for data in loader:
                    for info, soh, mode in data:
                        feat = model.encode_current(info.cuda(), mode.cuda())
                        features = torch.concat((features, feat.cpu()), 0)
                        bat_life = soh[:, 0, 0].unsqueeze(-1) * std + mean
                        labels = torch.concat((labels, bat_life.cpu()), 0)
            if state == "test":
                for data in loader:
                    info, soh, mode = data
                    feat = model.encode_current(info.cuda(), mode.cuda())
                    features = torch.concat((features, feat.cpu()), 0)
                    bat_life = soh[:, 0, 0].unsqueeze(-1)
                    labels = torch.concat((labels, bat_life.cpu()), 0)
        return features, labels
    X_train, y_train = extract_features(train_loader, state="train")
    X_test, y_test = extract_features(test_loader, state="test")
    # 简单线性回归验证
    clf = LinearRegression().fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc


def normal_evaluation(model,train_loader, test_loader):
    model.eval()
    prediction, real = torch.empty(0), torch.empty(0)
    std = train_loader.dataset.std_life
    mean = train_loader.dataset.mean_life
    for data in test_loader:
        info, soh, mode = data
        output = model(info.cuda(), soh.cuda(), mode.cuda())
        life_prediction, bat_life = output["supervised"]
        prediction = torch.concat((prediction, life_prediction.cpu() * std + mean), 0)
        real = torch.concat((real, bat_life.cpu()), 0)
    return prediction, real


class BatteryAugmentation:
    def __init__(self,noise_scale=0.01, shift_range=10, scale_range=(0.95, 1.05)):
        self.noise_scale = noise_scale  # 模拟测量噪声
        self.shift_range = shift_range  # 循环偏移（±5圈）
        self.scale_range = scale_range  # 容量缩放（模拟不同初始容量）

    def __call__(self, sequence, type=None):
        life = 0
        seq = sequence
        if type == "label":
            seq = sequence[:, 1:]
            life = sequence[:, 0].unsqueeze(-1)
        # 1. 随机缩放（模拟电池个体差异）
        scale = np.random.uniform(*self.scale_range)
        seq = seq * scale
        if type == "label" or type == "input":
            # 2. 添加高斯噪声（模拟传感器误差）
            noise = np.random.normal(0, self.noise_scale, seq.shape)
            seq = seq + noise
            # if type == "label":
            #     seq = np.maximum(seq, 0)
        # 3. 随机时间平移（对循环轴）
        shift = np.random.randint(-self.shift_range, self.shift_range)
        if type == "input":
            if shift > 0:
                seq = np.pad(seq, ((0, 0), (shift, 0), (shift, 0)), mode='edge')[:, :-shift, :-shift]
            elif shift < 0:
                seq = np.pad(seq, ((0, 0), (0, -shift), (0, -shift)), mode='edge')[:, -shift:, -shift:]
        if type == "label":
            if shift > 0:
                seq = np.pad(seq, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
            elif shift < 0:
                seq = np.pad(seq, ((0, 0), (0, -shift)), mode='edge')[:, -shift:]
            seq = np.concat((life+shift/seq.shape[1], seq), 1)
        return torch.as_tensor(seq).to(torch.float32)


# 在 Dataset 中使用
class ContrastiveBatteryDataset(dataset.Dataset):
    def __init__(self, f, q, m):
        super(ContrastiveBatteryDataset, self).__init__()
        augment = True
        self.feature = f
        self.capacity = q
        self.mode = m
        self.mean_life = torch.mean(q[:, 0, 0])
        self.std_life = torch.std(q[:, 0, 0])
        # 直接预测寿命需要归一化
        self.capacity[:, :, 0] = (self.capacity[:, :, 0] - self.mean_life) / self.std_life
        self.augment = BatteryAugmentation() if augment else None


    def __getitem__(self, index):
        input = self.feature[index]
        label = self.capacity[index]
        mode = self.mode[index]
        # 为对比学习生成两个不同视图
        if self.augment:
            input_v1 = self.augment(input, type="input")
            input_v2 = self.augment(input, type="input")
            label_v1 = self.augment(label, type="label")
            label_v2 = self.augment(label, type="label")
            mode_v1 =  self.augment(mode, type="mode")
            mode_v2 = self.augment(mode, type="mode")
            return (input, label, mode), (input_v1, label_v1, mode_v1), (input_v2, label_v2, mode_v2)

        return input, label, mode

    def __len__(self):
        return len(self.feature)