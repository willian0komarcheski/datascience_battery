from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from config import path_input as path

#se for necessario mudar os dados de treinamento precisa somente mudar
#esse arquivo e chamar a nova funcao criada como "get_exp_based_df" que o resto funcionara normalmente, claro, prestando bastante atenção aos limites do tensorflow e  do array numpy

#processamento de dados para usar na rede neural

def to_padded_numpy(l, shape):
    padded_array = np.zeros(shape)
    padded_array[:len(l)] = l
    return padded_array


def preprocess_data_to_cycles():
    dis = os.listdir(path)
    dis_mat = []
    battery_grp = {}

    for i in dis:
        filtered_list = list(filter(lambda x: x.split(
            '.')[-1] == 'mat', os.listdir(f"{path}/{i}")))
        battery_grp[i.split('BatteryAgingARC')[-1][1:]
                    ] = list(map(lambda x: x.split('.')[0], filtered_list))
        dis_mat.extend(list(map(lambda x: f"{path}/{i}/{x}", filtered_list)))

    battery_grp['5_6_7_18_25_26_27_28_29_30_31_32_33_34_36_38_39_40_41_42_43_44_45_46_47_48_49_50_51_52_53_54_55_56'] = battery_grp['ALL']
    del battery_grp['ALL']

    bs = [x.split('/')[-1].split('.')[0] for x in dis_mat]

    ds = []
    for b in dis_mat:
        ds.append(loadmat(b))

    types = []
    times = []
    ambient_temperatures = []
    datas = []

    for i in range(len(ds)):
        x = ds[i][bs[i]]["cycle"][0][0][0]
        ambient_temperatures.append(
            list(map(lambda y: y[0][0], x['ambient_temperature'])))
        types.append(x['type'])
        times.append(x['time'])
        datas.append(x['data'])

    batteries = []
    cycles = []
    for i in range(len(ds)):
        batteries.append(bs[i])
        cycles.append(datas[i].size)

    battery_cycle_df = pd.DataFrame(
        {'Battery': batteries, 'Cycle': cycles}).sort_values('Battery', ascending=True)
    battery_cycle_df.drop_duplicates(inplace=True)

    Cycles = {}
    params = ['Voltage_measured', 'Current_measured', 'Temperature_measured',
              'Current_load', 'Voltage_load', 'Time', 'Capacity', ]

    for i in range(len(bs)):
        Cycles[bs[i]] = {}
        Cycles[bs[i]]['count'] = 0
        for param in params:
            Cycles[bs[i]][param] = []
            for j in range(datas[i].size):
                if types[i][j] == 'discharge':
                    Cycles[bs[i]][param].append(datas[i][j][param][0][0][0])

        cap = []
        amb_temp = []
        for j in range(datas[i].size):
            if types[i][j] == 'discharge':
                cap.append(datas[i][j]['Capacity'][0][0][0])
                amb_temp.append(ambient_temperatures[i][j])

        Cycles[bs[i]]['Capacity'] = np.array(cap)
        Cycles[bs[i]]['ambient_temperatures'] = np.array(amb_temp)
    Cycles = pd.DataFrame(Cycles)

    return Cycles


def get_exp_based_df(exp):
    Cycles = preprocess_data_to_cycles()
    df_all = pd.DataFrame({})
    max_len = 0

    exp_try_out = exp

    for bat in exp_try_out:
        df = pd.DataFrame({})
        cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                'Current_load', 'Voltage_load', 'Time', 'Capacity', 'ambient_temperatures']
        for col in cols:
            df[col] = Cycles[bat][col]
        max_l = np.max(df['Time'].apply(lambda x: len(x)).values)
        max_len = max(max_l, max_len)
        df_all = pd.concat([df_all, df], ignore_index=True)

    df = df_all.reset_index(drop=True)
    df

    for i, j in enumerate(df['Capacity']):
        try:
            if len(j):
                df['Capacity'][i] = j[0]
            else:
                df['Capacity'][i] = 0
        except:
            pass

    df_x = df.drop(columns=['Capacity', 'ambient_temperatures']).values
    df_y = df['Capacity'].values

    n, m = df_x.shape[0], df_x.shape[1]
    temp2 = np.zeros((n, m, max_len))
    for i in range(n):
        for j in range(m):
            temp2[i][j] = to_padded_numpy(df_x[i][j], max_len)

    df_x = temp2
    return df_x, df_y