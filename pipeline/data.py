"""Bearing 1"""

import pandas as pd
import numpy as np
import os
from .utils import PrepareData
from sklearn.preprocessing import MinMaxScaler


def get_initial_input():

    path = 'data_condition_1/'
    data = os.listdir(path)

    for i in data:
        globals()[f"{i}"] = pd.read_csv(path+i).drop('Unnamed: 0', axis=1)

    # train
    array1_1, rul1_1 = PrepareData.get_spectogram_array(
        data_Bearing1_1, 75, 'Horiz_accel', 'rul')
    array1_2, rul1_2 = PrepareData.get_spectogram_array(
        data_Bearing1_2, 75, 'Horiz_accel', 'rul')
    array1_5, rul1_5 = PrepareData.get_spectogram_array(
        data_Bearing1_5, 75, 'Horiz_accel', 'rul')
    array1_6, rul1_6 = PrepareData.get_spectogram_array(
        data_Bearing1_6, 75, 'Horiz_accel', 'rul')

    # test
    array1_3, rul1_3 = PrepareData.get_spectogram_array(
        data_Bearing1_3, 75, 'Horiz_accel', 'rul')
    array1_4, rul1_4 = PrepareData.get_spectogram_array(
        data_Bearing1_4, 75, 'Horiz_accel', 'rul')
    array1_7, rul1_7 = PrepareData.get_spectogram_array(
        data_Bearing1_7, 75, 'Horiz_accel', 'rul')

    # train
    train = np.concatenate((array1_7, array1_2, array1_5, array1_1, array1_3, array1_4), axis=0)
    label_train = np.concatenate((rul1_7, rul1_2, rul1_5, rul1_1, rul1_3, rul1_4), axis=0)

    # test
    test = array1_6#np.concatenate((array1_3, array1_4, array1_7), axis=0)
    label_test = rul1_6#np.concatenate((rul1_3, rul1_4, rul1_7), axis=0)

    # reshape train
    train = train.reshape(
        (train.shape[0],
         train.shape[1],
         train.shape[2],
         1))

    # reshape test
    test = test.reshape(
        (test.shape[0],
         test.shape[1],
         test.shape[2],
         1))

    # SCALER
    #train_df = pd.DataFrame(data=train.reshape(8585, -1))
    #test_df = pd.DataFrame(data=test.reshape(6062, -1))

    ## Adicione as colunas de rótulos aos DataFrames
    #train_df['Label'] = label_train
    #test_df['Label'] = label_test

    ## Adicione a coluna 'type_of_data' para indicar se é de treino ou teste
    #train_df['type_of_data'] = 'train'
    #test_df['type_of_data'] = 'test'

    ## # Concatene os DataFrames
    #combined_df = pd.concat([train_df, test_df])

    ## # Crie um Min-Max Scaler
    #scaler = MinMaxScaler()
    ## #
    ## # Ajuste o scaler apenas às labels
    #combined_df['Label'] = scaler.fit_transform(
    #    combined_df['Label'].values.reshape(-1, 1))
    #
    ## # Separe os dados de volta em treino e teste
    #train = combined_df[combined_df['type_of_data'] ==
    #                    'train'].iloc[:, :-2].values.reshape(8585, 38, 38, 1)
    #test = combined_df[combined_df['type_of_data'] ==
    #                   'test'].iloc[:, :-2].values.reshape(6062, 38, 38, 1)
    #label_train = combined_df[combined_df['type_of_data']
    #                          == 'train']['Label'].values
    #label_test = combined_df[combined_df['type_of_data']
    #                         == 'test']['Label'].values

    return train, test, label_train, label_test
