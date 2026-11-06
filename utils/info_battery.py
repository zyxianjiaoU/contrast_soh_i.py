import math
import os

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d


# from scipy.io import loadmat
# from openpyxl import load_workbook
# from matplotlib import pyplot as plt
# import warnings
# warnings.filterwarnings('error')


def read_xls(sheet, Header):
    #按行保存单元格值((1,2,3),(4,5,6),(7,8,9))
    valuesByRow = tuple(sheet.values)
    # 第一行，也就是表头
    firstRow = valuesByRow[0]
    # 空列表
    cellValueList = []
    for i, v in enumerate(firstRow):
        if v == Header:
            for row in valuesByRow[1:]:
                cellValueList.append(row[i])
            break
    return np.array(cellValueList)


def get_soh_from_q(qc, qd, l_soh, soh_step, bat_life):
    length = int(l_soh / soh_step)
    assert length == len(np.arange(0, l_soh, step=soh_step))
    qc = np.round(qc, decimals=4)
    qd = np.round(qd, decimals=4)
    soh_c = qc / qc[0]
    soh_d = qd / qd[0]
    soh = np.zeros((1, 2, length))
    l = len(np.arange(0, len(soh_c), step=soh_step))
    if l_soh  > len(soh_c):
        coef_qc = np.polyfit(np.arange(len(soh_c) - 50, len(soh_c)), soh_c[-50:], 1)
        coef_qd = np.polyfit(np.arange(len(soh_d) - 50, len(soh_d)), soh_d[-50:], 1)
        x_fit = np.arange(len(soh_c), l_soh)
        qc_fit = np.polyval(coef_qc, x_fit)
        qd_fit = np.polyval(coef_qd, x_fit)
        soh_c = np.concat((soh_c, qc_fit))
        soh_d = np.concat((soh_d, qd_fit))
        soh[0][0] = soh_c[np.arange(0, len(soh_c), step=soh_step)]
        soh[0][1] = soh_d[np.arange(0, len(soh_d), step=soh_step)]
    else:
        soh[0][0] = soh_c[np.arange(0, l_soh, step=soh_step)]
        soh[0][1] = soh_d[np.arange(0, l_soh, step=soh_step)]
    soh[0][0][0] = bat_life
    soh[0][1][0] = bat_life
    return soh


def data_deposit(voltage, q,length):
    # 去除相同的U点
    repeat = np.zeros(0)
    for a in range(1, len(voltage) - 1):
        if voltage[a] == voltage[a + 1]:
            q[a + 1] = 2 * q[a] - q[a - 1]
            repeat = np.append(repeat, a)
    final = np.setdiff1d(np.arange(len(voltage)), repeat)
    voltage, q = voltage[final], q[final]
    U_inter = np.linspace(start=2, stop=3.5, num=length)
    f_u = interp1d(voltage, q, 'linear', fill_value="extrapolate")
    q_new = f_u(U_inter)
    return q_new


class mode_analysis:
    def __init__(self,  q, current, voltage, discharge_start, split=10):
        self.i = current
        self.u = voltage
        self.q = q
        self.discharge_start = discharge_start
        self.split = split

    def _charge_discharge_mode(self, state:str):
        mode_per_bat = np.zeros((8, self.split))
        if state == 'charge':
            i, u, q = self.i[:self.discharge_start], self.u[:self.discharge_start], self.q[:self.discharge_start]
        elif state == 'discharge':
            i, u, q = self.i[self.discharge_start:], self.u[self.discharge_start:], self.q[self.discharge_start:]
        else:
            raise Exception('{} not supported'.format(state))
        assert q[0] < 0.1 and len(q) > 50
        q_c = np.float16(max(q))
        for k in range(0, self.split):
            soc = np.logical_and(k * q_c/ self.split < q, q < (k + 1) * q_c / self.split)
            mode_per_bat[0][k] = np.mean(i[soc])
            mode_per_bat[1][k] = np.var(i[soc])
            mode_per_bat[2][k] = pd.Series(i).kurt()
            mode_per_bat[3][k] = pd.Series(i).skew()
            mode_per_bat[4][k] = np.mean(u[soc])
            mode_per_bat[5][k] = np.var(u[soc])
            mode_per_bat[6][k] = pd.Series(u).kurt()
            mode_per_bat[7][k] = pd.Series(u).skew()
        return mode_per_bat


    def mode(self):
        charge_mode = self._charge_discharge_mode('charge')
        discharge_mode = self._charge_discharge_mode('discharge')
        return charge_mode, discharge_mode


# def hust_data(info_length, l_soh, soh_step):
#     """
#     A dataset with 77 LFP/graphite cells (1.1 Ah nominal capacity and 3.3 V nominal voltage).
#     The cells were cycled with an identical charge protocol
#     but different multi-stage discharge protocols at a constant temperature of 30°C.
#     (a) An identical charge protocol C1 (5C, 0% SOC-80% SOC)-C2 (1C, 80% SOC-the first 3.6 V)   c1=5C, c2=1C
#     followed by a constant-voltage charge (3.6 V-C/20) for cells. (b) Four-step discharge from 100% SOC to
#     a cutoff voltage of 2 V. Diverse discharge protocols C1 (100% SOC-60% SOC)-C2 (60% SOC-40% SOC)-C3
#     (40% SOC-20% SOC)-C4 (20% SOC-2 V) for different cells, where C1-C4 represent constant-current
#     discharge rates at the four steps, respectively.
#     """
#     path = 'hust_data/our_data'
#     data_path = os.listdir(path)
#     c, info, mode_info, soh_info = 0, 0, 0, 0
#     for bat_idx in range(len(data_path)):
#         battery_path = os.path.join(path, data_path[bat_idx])
#         with open(battery_path, 'rb') as f:
#             data = pickle.load(f)
#             data_bat = data[list(data.keys())[0]]
#             battery = data_bat['data']
#             length = int(l_soh / soh_step)
#             assert length == len(np.arange(0, l_soh, soh_step))
#             soh = np.zeros((1, 2, length))
#             info_each_bat = np.zeros((1, 100, 3, info_length))
#             mode_per_bat = np.zeros((1, 100, 2, 2, 10))
#             # battery.keys(): dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,...])
#             assert list(battery.keys())[0]==1
#             for cycle_idx in sorted(list(battery.keys()))[:100]:  #   1, 2, 3, 4, 5, 6, 7, 8, 9, 10,...
#                 cycle = cycle_idx-1
#                 data_per_cycle = battery[cycle_idx]
#                 # data_per_cycle.keys()
#                 # Index(['Status', 'Cycle number', 'Current (mA)', 'Voltage (V)', 'Capacity (mAh)', 'Time (s)'], dtype='object')
#                 # status = data_per_cycle['Status']
#                 cycle_number = data_per_cycle['Cycle number'].drop_duplicates()
#                 assert len(cycle_number) == 1
#                 current = (np.array(data_per_cycle['Current (mA)'] / 1000)).round(4)
#                 voltage = np.array(data_per_cycle['Voltage (V)']).round(4)
#                 times = (np.array(data_per_cycle['Time (s)']) / 60).round(6)
#                 q = (np.array(data_per_cycle['Capacity (mAh)']) / 1000).round(4)
#                 q_max = max(q)
#                 position_max_q = np.where(q == max(q))[0][0]
#                 q[(position_max_q+1):] = q_max - q[(position_max_q+1):]
#                 # P1 INFO
#                 info_each_bat = data_deposit(info_each_bat, times, current, voltage, q, 30, cycle)
#                 # P2 SOH
#                 # get soh of charging and discharging
#                 if cycle < position_max_q:
#                     soh[0][0][cycle] = q_max
#                 else:
#                     soh[0][1][cycle] = q[-1]
#                 # P3 MODE ANALYSIS
#                 # assert len(qc) == position_max_q
#                 assert q[position_max_q] - q[position_max_q + 1] > 0.5
#                 mode = mode_analysis(times, q, current, voltage, position_max_q)
#                 #   mode_per_bat = np.zeros((2, self.split))
#                 mode_per_bat[0][cycle][0], mode_per_bat[0][cycle][1] = mode.mode()
#             if max(battery.keys()) < l_soh + 1:     # fill soh
#                 soh[0][0][max(battery.keys()):] = q_max
#                 soh[0][1][max(battery.keys()):] = q[-1]
#             info = cat(info, info_each_bat)
#             mode_info = cat(mode_info, mode_per_bat)
#             soh_info = cat(soh_info, soh)
#             c += 1
#     return info, mode_info, soh_info
#
#
# def kti_data(info_length, l_soh, soh_step):
#     path = 'dataset3'
#     data_path = os.listdir(path)
#     info, mode_info, soh_info = 0, 0, 0
#     for batch_idx in range(len(data_path)):
#         if '.pdf' in data_path[batch_idx]:
#             continue
#         path1 = os.path.join(path, data_path[batch_idx])
#         path2 = os.listdir(path1)
#         for bat_idx in range(len(path2)):
#             battery_path = os.path.join(path1, path2[bat_idx])
#             data = pd.read_csv(battery_path, delimiter=',')
#             cycle = np.array(data['cycle number'].tolist(), dtype=np.int64)
#             all_cycle = sorted(np.unique(cycle))
#             if len(all_cycle) < 110:
#                 continue
#             temperature = int(re.findall(r'CY(\d+)-', path2[bat_idx])[0])
#             times = (np.array(data['time/s'].tolist()) / 60).round(decimals=6) # min as unit
#             voltage = np.array(data['Ecell/V'].tolist()).round(decimals=4)  # V as unit
#             current = (np.array(data['<I>/mA'].tolist()) / 1000).round(decimals=4) # A as unit
#             qd = (np.array(data['Q discharge/mA.h'].tolist()) / 1000).round(decimals=4) # Ah as unit
#             qc = (np.array(data['Q charge/mA.h'].tolist()) / 1000).round(decimals=4)
#
#             info_each_bat = np.zeros((1, 100, 3, info_length))
#             mode_per_bat = np.zeros((1, 100, 2, 2, 10))
#             # current_control = data['control/mA'].tolist() / 1000
#             # P1 SOH
#             # get soh of charging and discharging
#             length = int(l_soh / soh_step)
#             assert length == len(np.arange(2, l_soh + 2, soh_step))
#             soh = np.zeros((1, 2, length))
#             idx = 0
#             for each_cycle in np.arange(2, l_soh + 2, soh_step):
#                 if each_cycle < max(all_cycle):
#                     soh[0][0][idx] = max(qc[cycle==each_cycle])
#                     soh[0][1][idx] = max(qd[cycle==each_cycle])
#                 else:
#                     soh[0][0][idx] = soh[0][0][idx-1]
#                     soh[0][1][idx] = soh[0][1][idx-1]
#                 idx += 1
#             c = 0
#             for cycle_idx in range(110):
#                 position = (cycle == all_cycle[cycle_idx])
#                 t, u, i, qc_per_cycle, qd_per_cycle = times[position], voltage[position], current[position], qc[position], qd[position]
#                 t -= t[0]
#                 if len(np.where(i < 0)[0]) == 0:
#                     continue
#                 duplic = [k for k in range(len(t)-1) if t[k] >= t[k+1]]
#                 t, u, i, qc_per_cycle, qd_per_cycle = (np.delete(t, duplic), np.delete(u, duplic), np.delete(i, duplic),
#                                                        np.delete(qc_per_cycle, duplic), np.delete(qd_per_cycle, duplic))
#                 discharge_start = np.where(i < 0)[0][0]
#                 q = np.concatenate((qc_per_cycle[:discharge_start], qd_per_cycle[discharge_start:]))
#                 assert q[discharge_start-1] - q[discharge_start] > 0.5
#                 # data_deposit(info_each_bat, times, current, voltage, q, length, cycle)
#                 # P2 INFO 100
#                 info_each_bat = data_deposit(info_each_bat, t, i, u, q, temperature, c)
#                 # P3 MODE
#                 mode = mode_analysis(t, q, i, u, discharge_start)
#                 mode_per_bat[0][c][0], mode_per_bat[0][c][1] = mode.mode()
#                 c += 1
#                 if c == 100:
#                     break
#             info = cat(info, info_each_bat)
#             mode_info = cat(mode_info, mode_per_bat)
#             soh_info = cat(soh_info, soh)
#     return info, mode_info, soh_info
#
#
# def calce_data(info_length, l_soh, soh_step):
#     length = int(l_soh / soh_step)
#     assert length == len(np.arange(0, l_soh, soh_step))
#     path = 'calce_data'
#     data_path = sorted(os.listdir(path), key=lambda s: int(re.findall(r"^(\d+)~", s)[0]))
#     bat_dict= {}
#     soh_info = np.zeros((1000, 2, length))
#     info = np.zeros((1000, 100, 3, info_length))
#     mode_info = np.zeros((1000, 100, 2, 2, 10))
#     drop = np.array([])
#     for batch_idx in range(len(data_path)):
#         path1 = os.path.join(path, data_path[batch_idx])
#         path2 = os.listdir(path1)
#         for bat_idx in range(len(path2)):
#             battery_path = os.path.join(path1, path2[bat_idx])
#             data = load_workbook(battery_path)
#             sheet_name = re.split(r'-', path2[bat_idx])
#             cycle_min = int(sheet_name[1])
#             cycle_max = int(sheet_name[2])
#             T = int(re.findall(r'-(\d+)DU', path2[bat_idx])[0])
#             assert cycle_max - cycle_min == 49
#             # sheet_id  = sheet_name[3]
#             for sheet in data.worksheets:
#                 if sheet.title == 'Global_Info':
#                     continue
#                 ch_id = re.findall(r'_(\d+)', sheet.title)[0]
#                 if ch_id not in bat_dict.keys():
#                     bat_dict[ch_id] = len(bat_dict.keys())
#                 if 'Statistics' in sheet.title and 'ByStep' not in sheet.title:
#                     qc = read_xls(sheet, 'Charge_Capacity(Ah)').round(decimals=4)
#                     qd = read_xls(sheet, 'Discharge_Capacity(Ah)').round(decimals=4)
#                     cycle_idx = read_xls(sheet, 'Cycle_Index').astype(int)
#                     # P1 SOH
#                     # get soh of charging and discharging in this 50 cycles
#                     if len(cycle_idx) > 50:
#                         cycle_idx = cycle_idx[:50]
#                     for k in cycle_idx:
#                         if (k + cycle_min - 2) in np.arange(0, l_soh, soh_step):
#                             idx = np.where(soh_info[bat_dict[ch_id]][0]==0)[0][0]
#                             # assert qc[idx] != 0 and qd[idx] != 0
#                             if qc [k-1] < 0.5 and k > 1:
#                                 qc[k-1] = qc[k-2]
#                             if qd [k-1] < 0.5 and k > 1:
#                                 qd[k-1] = qc[k-2]
#                             soh_info[bat_dict[ch_id]][0][idx] = qc[k-1]
#                             soh_info[bat_dict[ch_id]][1][idx] = qd[k-1]
#                     if max(cycle_idx) < 50 and cycle_max < 101:
#                         drop = np.append(drop, bat_dict[ch_id])
#                         continue
#                 if 'Channel' in sheet.title and '_Channel_' not in sheet.title:
#                     if cycle_max > 100:
#                         continue
#                     cycle_idx = read_xls(sheet, 'Cycle_Index').astype(int)
#                     step_idx = read_xls(sheet, 'Step_Index').astype(int)
#                     voltage = read_xls(sheet, 'Voltage(V)').round(decimals=4)
#                     current = read_xls(sheet, 'Current(A)').round(decimals=4)
#                     time = read_xls(sheet, 'Test_Time(s)').round(decimals=6) / 60
#                     qc_cycles = read_xls(sheet, 'Charge_Capacity(Ah)')
#                     qd_cycles = read_xls(sheet, 'Discharge_Capacity(Ah)')
#                     if max(cycle_idx) < 50:
#                         continue
#                     info_each_bat = np.zeros((1, 50, 3, info_length))
#                     pos_step = step_idx < 8
#                     for k in range(50):
#                         # data_deposit(info_each_bat, time, current, voltage, q, cycle)
#                         pos = np.logical_and(cycle_idx == k+1, pos_step)
#                         t, i, u, qc_per_cycle, qd_per_cycle = (time[pos], current[pos], voltage[pos], qc_cycles[pos], qd_cycles[pos])
#                         t = t - t[0]
#                         cycle_actual = k + cycle_min - 1
#                         if max(qc_per_cycle) < 0.1:
#                             if k > 0:
#                                 info_each_bat[0][k] = info_each_bat[0][k-1]
#                                 mode_info[bat_dict[ch_id]][cycle_actual] = mode_info[bat_dict[ch_id]][cycle_actual-1]
#                             continue
#                         discharge_start = np.where(step_idx[pos] == 5)[0][0] - 2
#                         q = np.concatenate((qc_per_cycle[:discharge_start], qd_per_cycle[discharge_start:]))
#                         if i[discharge_start] != 0 and i[discharge_start + 1] != 0:
#                             t = np.insert(t, discharge_start + 1, (t[discharge_start+1]+t[discharge_start])/2)
#                             i = np.insert(i, discharge_start + 1, 0)
#                             u = np.insert(u, discharge_start + 1, u[discharge_start+1])
#                             q = np.insert(q, discharge_start + 1, 0)
#                         info_each_bat = data_deposit(info_each_bat, t, i, u, q, T, k)
#                         mode = mode_analysis(t, q, i, u, discharge_start)
#                         mode_info[bat_dict[ch_id]][cycle_actual][0], mode_info[bat_dict[ch_id]][cycle_actual][1] = mode.mode()
#                     info[bat_dict[ch_id]][(cycle_min-1):cycle_max] = info_each_bat.squeeze(0)
#     bat_num = [i for i in range(soh_info.shape[0]) if soh_info[i][0][0] == 0]
#     bat_num = np.append(bat_num, drop)
#     bat_num = np.array(bat_num, dtype=int)
#     for a in range(soh_info.shape[0]):
#         for b in range(2):
#             l = np.where(soh_info[a][b]==0)[0]
#             if len(l) > 0:
#                 soh_info[a][b][l[0]:] = soh_info[a][b][l[0]-1]
#     soh_info, info, mode_info = (np.delete(soh_info, bat_num, axis=0), np.delete(info, bat_num, axis=0),
#                                  np.delete(mode_info, bat_num, axis=0))
#     return info, mode_info, soh_info


def severson_data(info_length:int, l_soh:int, soh_step:int):
    info, soh_info, mode_info = np.empty((0, 2, 100, info_length)), np.empty((0, 2, int(l_soh/soh_step))), np.empty((0, 100, 2, 8, 10))
    path = r'D:\battery capability\ddpm\dataset'
    data_paths = os.listdir(path)
    for batch_idx in range(len(data_paths)):
        batch_path = os.path.join(path, data_paths[batch_idx])
        data = h5py.File(batch_path)
        for bat_idx in range(len(data['batch/policy_readable'])):
            # P1 SOH
            # get soh of charging and discharging
            qc = np.array(data[data['batch/summary'][bat_idx][0]]['QCharge']).squeeze(0)[4:]
            qd = np.array(data[data['batch/summary'][bat_idx][0]]['QDischarge']).squeeze(0)[4:]
            bat_life = np.array(data[data['batch/cycle_life'][bat_idx][0]])[0][0]
            if math.isnan(bat_life):
                bat_life = len(qc)
            soh = get_soh_from_q(qc, qd, l_soh, soh_step, bat_life=int(bat_life))       # shape: (1, 2, length)
            # get i / u / mode info from first 100 cycle
            info_each_bat = np.zeros((1, 2, 100, info_length))
            mode_per_bat = np.zeros((1, 100, 2, 8, 10))
            for cycle in range(5, 105):
                voltage = np.array(data[data[data['batch/cycles'][bat_idx][0]]['V'][cycle][0]]).squeeze(0)
                current = np.array(data[data[data['batch/cycles'][bat_idx][0]]['I'][cycle][0]]).squeeze(0)
                qc_per_cycle = np.array(data[data[data['batch/cycles'][bat_idx][0]]['Qc'][cycle][0]]).squeeze(0)
                qd_per_cycle = np.array(data[data[data['batch/cycles'][bat_idx][0]]['Qd'][cycle][0]]).squeeze(0)
                discharge_start = np.where(current<-0.01)[0][0]
                # 取 充 放 电 过 程
                q = np.concatenate((qc_per_cycle[:discharge_start], qd_per_cycle[discharge_start:]))
                u_d = voltage[discharge_start:]
                qd = qd_per_cycle[discharge_start:]
                u_c = voltage[:discharge_start]
                qc = qd_per_cycle[:discharge_start]
                # P2 INFO 100
                info_each_bat[0][0][cycle-5] = data_deposit(voltage=u_d, q=qd, length=info_length)
                info_each_bat[0][1][cycle-5] = data_deposit(voltage=u_c, q=qc, length=info_length)
                # P3 MODE
                try:
                    assert q[discharge_start-1] - q[discharge_start] > 0.5
                except:
                    print(batch_path, bat_idx, cycle)
                mode = mode_analysis(q, current, voltage, discharge_start)
                # 充放电模式
                mode_per_bat[0][cycle-5][0], mode_per_bat[0][cycle-5][1] = mode.mode()
            info = np.concat((info, info_each_bat), 0)
            mode_info = np.concat((mode_info, mode_per_bat), 0)
            soh_info = np.concat((soh_info, soh), 0)
    return info, mode_info, soh_info


# def xjtu_data(info_length, l_soh, soh_step):
#     info, mode_info, soh_info = 0, 0, 0
#     for i in range(6):  # 6 batch
#         path = 'Battery Dataset/Batch-{}'.format(i + 1)
#         data_paths = os.listdir(path)
#         for datapath in data_paths:
#             battery_path = os.path.join(path, datapath)
#             data = loadmat(battery_path)
#             battery = data['data']
#             # P1 SOH
#             # get soh of charging
#             # qc and qd is total fading curve
#             qc = data['summary'][0][0]['charge_capacity_Ah'].squeeze(-1)
#             qd = data['summary'][0][0]['discharge_capacity_Ah'].squeeze(-1)
#             soh = get_soh_from_q(qc, qd, l_soh, soh_step)
#             # P2 AND P3 INFO 100 AND MODE
#             # get i / u / mode info
#             info_each_bat = np.zeros((1, 100, 3, info_length))
#             mode_per_bat = np.zeros((1, 100, 2, 2, 10))
#             for cycle in range(100):
#                 voltage = battery[0][cycle]['voltage_V'].squeeze(-1).round(decimals=4)
#                 current = battery[0][cycle]['current_A'].squeeze(-1).round(decimals=4)
#                 time = battery[0][cycle]['relative_time_min'].squeeze(-1).round(decimals=6)
#                 T = np.mean(battery[0][cycle]['temperature_C'].squeeze(-1)).round(decimals=2)
#                 q = battery[0][cycle]['capacity_Ah'].squeeze(-1).round(decimals=4)     # include discharge and charge
#                 # solve time gap
#                 pos0 = np.where(np.diff(time) < 0)[0] + 1
#                 if len(pos0) > 0:
#                     for k  in range(len(pos0)-1):
#                         time[pos0[k]:pos0[k+1]] += time[pos0[k]-1]
#                     time[pos0[-1]:] += time[pos0[-1]-1]
#                     voltage, current, time, q = np.delete(voltage, pos0), np.delete(current, pos0), np.delete(time, pos0), np.delete(q, pos0)
#                     try:
#                         assert len(np.where(np.diff(time) < 0)[0]) == 0
#                     except:
#                         print(battery_path, cycle, 'pos0')
#                 # data_deposit(info_each_bat, times, current, voltage, q, length, cycle)
#                 position_max_q = np.where(np.diff(q) < 0)[0][0]  # get 1st big gap of q
#                 info_each_bat = data_deposit(info_each_bat, time, current, voltage, q, T, cycle)
#                 # P3 MODE ANALYSIS
#                 # assert len(qc) == position_max_q
#                 mode = mode_analysis(time, q, current, voltage, position_max_q)
#                 mode_per_bat[0][cycle][0], mode_per_bat[0][cycle][1] = mode.mode()
#             info = cat(info, info_each_bat)
#             mode_info = cat(mode_info, mode_per_bat)
#             soh_info = cat(soh_info, soh)
#     return info, mode_info, soh_info
#
#
# def stanford_data(info_length, l_soh, soh_step):
#     info, mode_info, soh_info = 0, 0, 0
#     path_summary = 'stanford/aging_summary'
#     path_raw = 'stanford/raw_data'
#     data_path_1 = sorted(os.listdir(path_summary), key=lambda s: int(re.findall(r"_(\d+).csv", s)[0]))
#     data_path_2 = sorted(os.listdir(path_raw), key=lambda s: int(re.findall(r"_(\d+).csv", s)[0]))
#     for bat_idx in range(len(data_path_1)):
#         assert re.findall(r"_(\d+).csv", data_path_1[bat_idx])[0] == re.findall(r"_(\d+).csv", data_path_2[bat_idx])[0]
#         summary = os.path.join(path_summary, data_path_1[bat_idx])
#         raw = os.path.join(path_raw, data_path_2[bat_idx])
#         summary_data = pd.read_csv(summary, delimiter=',')
#         raw_data = pd.read_csv(raw, delimiter=',', dtype={'State': 'str', 'Full Step #':'object'})
#         # summary_data
#         qc = np.array(summary_data['Normalized Charge Capacity [-]'].tolist()).round(decimals=4)
#         qd = np.array(summary_data['Normalized Discharge Capacity [-]'].tolist()).round(decimals=4)
#         cycle = np.array(summary_data['Cycle'].tolist(), dtype=np.int64)
#         if len(cycle) < 100:
#             print(data_path_1[bat_idx])
#             continue
#         # P1 SOH
#         # get soh of charging and discharging
#         length = int(l_soh / soh_step)
#         assert length == len(np.arange(0, l_soh, soh_step))
#         soh = np.zeros((1, 2, length))
#         idx = 0
#         for each_cycle in np.arange(0, l_soh, soh_step):
#             if each_cycle < len(cycle):
#                 soh[0][0][idx] = qc[each_cycle]
#                 soh[0][1][idx] = qd[each_cycle]
#             else:
#                 soh[0][0][idx] = soh[0][0][idx - 1]
#                 soh[0][1][idx] = soh[0][1][idx - 1]
#             idx += 1
#         # cycle_data
#         times = (np.array(raw_data['Test (Sec)'].tolist()) / 60).round(decimals=6)  # min as unit
#         cycle_per_bat = np.array(raw_data['Cyc#'].tolist(), dtype=np.int64)
#         q_per_bat = np.abs(np.array(raw_data['Normalized Capacity (nominal capacity unit)'].tolist()).round(decimals=4))
#         voltage = np.array(raw_data['Volts'].tolist()).round(decimals=4)  # V as unit
#         c_rate = (np.array(raw_data['Normalized Current (C-rate)'].tolist())).round(decimals=4)  # A as unit
#         state = np.array(raw_data['State'].tolist())
#         info_each_bat = np.zeros((1, 100, 3, info_length))
#         mode_per_bat = np.zeros((1, 100, 2, 2, 10))
#         num = 0
#         for idx in cycle:
#             if num == 100:
#                 break
#             pos = cycle_per_bat == idx
#             t, q, u, i, s = times[pos], q_per_bat[pos], voltage[pos], c_rate[pos], state[pos]
#             t = t - t[0]
#             j = 0
#             if max(q) > 10 or 'C' not in s or 'D' not in s:
#                 continue
#             # P2 INFO 100
#             info_each_bat = data_deposit(info_each_bat, t, i, u, q, 35, num)
#             # P3 MODE
#             for k in range(len(s)-1):
#                 if (s[k] == 'C' and s[k+1] == 'D') or (s[k] == 'D' and s[k+1] == 'C'):
#                     j += 1
#             if j > 1:
#                 mode_per_bat[0][num] = -1
#                 num += 1
#                 continue
#             try:
#                 discharge_start = np.where(np.diff(q) < -0.5)[0][0]
#             except:
#                 print('kkk')
#             mode = mode_analysis(t, q, i, u, discharge_start)
#             mode_per_bat[0][num][0], mode_per_bat[0][num][1] = mode.mode()
#             num += 1
#         info = cat(info, info_each_bat)
#         mode_info = cat(mode_info, mode_per_bat)
#         soh_info = cat(soh_info, soh)
#     return info, mode_info, soh_info


def clip_data():
    info_resolution = 100
    l_soh = 1024
    step = 2
    # I_t：all cycles
    # info_each_cycle_6, mode_6, soh_6 = stanford_data(info_length=info_resolution, l_soh=l_soh, soh_step=step)
    # print('stanford over')
    # info_each_cycle_5, mode_5, soh_5 = calce_data(info_length=info_resolution, l_soh=l_soh, soh_step=step)
    # print('calce over')
    # info_each_cycle, mode, soh = xjtu_data(info_length=info_resolution, l_soh=l_soh, soh_step=step)
    # print('xjtu over')
    info_each_cycle, mode, soh = severson_data(info_length=info_resolution, l_soh=l_soh, soh_step=step)
    print('severson over')
    # info_each_cycle_3, mode_3, soh_3 = kti_data(info_length=info_resolution, l_soh=l_soh, soh_step=step)
    # print('kti over')
    # info_each_cycle_4, mode_4, soh_4 = hust_data(info_length=info_resolution, l_soh=l_soh, soh_step=step)
    # print('hust over')
    # cat
    # info_each_cycle = np.concatenate((info_each_cycle, info_each_cycle_2), 0)
    # mode = np.concatenate((mode, mode_2), 0)
    # soh = np.concatenate((soh, soh_2), 0)
    # # cat
    # info_each_cycle = np.concatenate((info_each_cycle, info_each_cycle_3), 0)
    # mode = np.concatenate((mode, mode_3), 0)
    # soh = np.concatenate((soh, soh_3), 0)
    # # cat
    # info_each_cycle = np.concatenate((info_each_cycle, info_each_cycle_4), 0)
    # mode = np.concatenate((mode, mode_4), 0)
    # soh = np.concatenate((soh, soh_4), 0)
    # # cat
    # info_each_cycle = np.concatenate((info_each_cycle, info_each_cycle_5), 0)
    # mode = np.concatenate((mode, mode_5), 0)
    # soh =np.concatenate((soh, soh_5), 0)
    # # cat
    # info_each_cycle = np.concatenate((info_each_cycle, info_each_cycle_6), 0)
    # mode = np.concatenate((mode, mode_6), 0)
    # soh =np.concatenate((soh, soh_6), 0)
    # shuffle per bat
    idx = np.arange(soh.shape[0])
    np.random.shuffle(idx)
    soh, info_each_cycle, mode = soh[idx], info_each_cycle[idx], mode[idx]
    train = int(0.8*soh.shape[0])
    soh_train, soh_test = torch.tensor(soh[:train]), torch.tensor(soh[train:])
    info_train, info_test = torch.tensor(info_each_cycle[:train]), torch.tensor(info_each_cycle[train:])
    mode_train, mode_test = torch.tensor(mode[:train]), torch.tensor(mode[train:])
    assert soh_train.shape[0] > soh_test.shape[0]

    data = {'soh_train':soh_train, 'soh_test':soh_test,
            'info_train':info_train,'info_test':info_test,
            'mode_train':mode_train, 'mode_test':mode_test}
    torch.save(data, 'clip_data_{}_{}.pt'.format(info_resolution, l_soh))

