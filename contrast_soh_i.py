# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import multiprocessing

import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from nn_model import memory_bank_contrastive_loss, MixFeature
from utils.nn import LoadPairs, linear_evaluation, ContrastiveBatteryDataset, normal_evaluation

# clip_data()
multiprocessing.freeze_support()
# device = "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load(r"D:\battery capability\ddpm\clip_data_100_1024.pt")
soh_train, soh_test = data['soh_train'].to(torch.float32), data['soh_test'].to(torch.float32)
info_train, info_test = data['info_train'].to(torch.float32), data['info_test'].to(torch.float32)
mode_train, mode_test = data['mode_train'].transpose(1, 2).to(torch.float32), data['mode_test'].transpose(1, 2).to(torch.float32)

bs = 16
l_rate = 0.001
# 对比学习学习率0.1~0.001
# 监督学习学习率0.01~0.001
training_cycle = 200
training_mode = "supervised"
model = MixFeature(training_mode=training_mode).to(device)
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
MseLoss = torch.nn.MSELoss()

# data_train = LoadPairs(info_train, soh_train, mode_train)
data_train = ContrastiveBatteryDataset(info_train, soh_train, mode_train)
load_data_train = DataLoader(data_train, shuffle=True, batch_size=bs)
data_test = LoadPairs(info_test, soh_test, mode_test)
# data_test = ContrastiveBatteryDataset(info_test, soh_test, mode_test)
load_data_test= DataLoader(data_test, shuffle=True, batch_size=bs)
optim = torch.optim.AdamW(model.parameters(), lr=l_rate, weight_decay=1e-4)
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
print('----------training start---------')
epoch = 0
for epoch in range(training_cycle):
    total_loss = 0
    j = 0
    # correct_train = 0
    model.train()
    # 训 练 网 络
    # for data_pair in load_data_train:
    for data_pairs in load_data_train:
        for data_pair in data_pairs:
            info, soh, mode = data_pair
            info, soh, mode = info.to(device), soh.to(device), mode.to(device)
            output = model(info, soh, mode)
            if training_mode == "supervised":
                prediction, bat_life = output[training_mode]
                loss = MseLoss(prediction, bat_life)
            elif training_mode == "contrastive":
                logits_soh, logits_current, labels = output[training_mode]
                loss = memory_bank_contrastive_loss(logits_current, logits_soh, labels)
            else:
                contra_output, supervised_output = output["contrastive"], output["supervised"]
                prediction, bat_life = supervised_output
                logits_soh, logits_current, labels = contra_output
                loss = MseLoss(prediction, bat_life) + memory_bank_contrastive_loss(logits_current, logits_soh, labels)
            total_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1E3, norm_type=1)
            optim.step()
            j += 1
            # 数据增强
            model.update_memory_banks(info, soh, mode)
    ExpLR.step()
    # 验 证
    # test
    test_acc, r2 = 0, 0
    l_r = l_rate
    if (epoch + 1) % 1 == 0:
        # # 检查梯度
        # soh_encoder_grad_norm = 0
        # info_grad_norm = 0
        # for name, param in model.named_parameters():
        #     if 'soh_encoder' in name:
        #         if param.grad is not None:
        #             soh_encoder_grad_norm += param.grad.norm().item() ** 2
        #     if 'encode_current' in name:
        #         if param.grad is not None:
        #             info_grad_norm += param.grad.norm().item() ** 2
        # print(f"Contrastive branch grad norm: {soh_encoder_grad_norm ** 0.5:.6f}")
        # print(f"info_grad_norm branch grad norm: {info_grad_norm ** 0.5:.6f}")
        l_r = optim.state_dict()['param_groups'][0]['lr']
        print('--------epoch (%d), average training loss: %.3f, l_r=%.6f--------' % (epoch + 1, total_loss/j, l_r))
        if training_mode == "contrastive" or training_mode == "hybrid":
            # 对比学习使用linear验证
            test_acc = linear_evaluation(model, load_data_train, load_data_test)
            print("---------contrastive accuracy: %.3f ----------" % test_acc)
            # 监督学习验证
        if training_mode == "supervised" or training_mode == "hybrid":
            prediction, real = normal_evaluation(model, load_data_train, load_data_test)
            r2 = r2_score(real.detach().numpy(), prediction.detach().numpy())
            print('---------supervised accuracy: %.3f ----------' % r2)
    if test_acc > 0.8 and training_mode == "contrastive":
        print('-----stop training----------')
        print('--------epoch (%d), average training loss: %.3f, l_r=%.6f--------' % (epoch + 1, total_loss/j, l_r))
        print("---------contrastive accuracy: %.3f ----------" % test_acc)
        break
    if r2 > 0.8 and training_mode == "supervised":
        print('-----stop training----------')
        print('--------epoch (%d), average training loss: %.3f, l_r=%.6f--------' % (epoch + 1, total_loss/j, l_r))
        print('---------supervised accuracy: %.3f ----------' % r2)
        break
    if test_acc > 0.7 and r2 > 0.7 and  training_mode == "hybrid":
        print('-----stop training----------')
        print('--------epoch (%d), average training loss: %.3f, l_r=%.6f--------' % (epoch + 1, total_loss/j, l_r))
        print('---------supervised accuracy: %.3f ----------' % r2)
        print("---------contrastive accuracy: %.3f ----------" % test_acc)
        break
checkpoint = {
    'model': model.state_dict(),
    'original lr':l_rate,
    'training_cycle':epoch,
    'batch_size':bs,
    "training_mode":training_mode,
}
torch.save(checkpoint, 'checkpoints/supervise/supervised_model.pt')
print("model saved!}")