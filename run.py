"""Bearing 1"""

import pprint
import pandas as pd
import numpy as np
import os
import tqdm
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from pipeline.utils import PrepareData, compare_plot
from pipeline import generate_model, get_initial_input
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error



def rejuste_index(lista_dfs):
    """Rejusta o index para o plot do RUL."""
    for index, df in enumerate(lista_dfs):
        if index == 0:
            df['index'] = df.index.tolist()
        else:
            shape_anterior = lista_dfs[index-1]['index'].tail(1).values[0]
            df['index'] = np.arange(
                shape_anterior, df.shape[0] + shape_anterior)
        df['PREDICTION'] = df['PREDICTION'].apply(lambda x: 0 if x < 0 else x)

def reajuste_predictions(lista_dfs):
    for df in lista_dfs:
        max_ = df['REAL'].max()
        df['PREDICTION'] = df['PREDICTION'].apply(lambda x: x if x <= max_ else max_)


for ep in tqdm.tqdm([50]):
#for ep in tqdm.tqdm([300]):
    epochs = ep
    train, test, label_train, label_test = get_initial_input()
    model2, history = generate_model(
        train=train, label_train=label_train, epochs=epochs, bearing=6)

    predict_rul_test = np.maximum(model2.predict(test), 0)
    predict_rul_train = np.maximum(model2.predict(train), 0)
    

    predict_rul_test = [i[0] for i in predict_rul_test.tolist()]
    predict_rul_train = [i[0] for i in predict_rul_train.tolist()]



    # Round dos valores preditos para valores inteiros.
    predict_rul_test = [round(rul) for rul in predict_rul_test]
    predict_rul_train = [round(rul) for rul in predict_rul_train]

    data_result_train = pd.DataFrame([predict_rul_train, label_train]).T
    data_result_test = pd.DataFrame([predict_rul_test, label_test]).T

    coluns = ['PREDICTION', 'REAL']
    data_result_train.columns = coluns
    data_result_test.columns = coluns

    min_train = data_result_train['REAL'].min()
    idx = list(
        data_result_train[data_result_train['REAL'] == min_train].index)# 0.10000000149011612].index)

    d1_train = data_result_train[idx[0]:idx[1]].sort_values(
        'REAL', ascending=False).reset_index(drop=True)
    d2_train = data_result_train[idx[1]:idx[2]].sort_values(
        'REAL', ascending=False).reset_index(drop=True)
    d3_train = data_result_train[idx[2]:idx[3]].sort_values(
        'REAL', ascending=False).reset_index(drop=True)
    d4_train = data_result_train[idx[3]:idx[4]].sort_values(
        'REAL', ascending=False).reset_index(drop=True)
    d5_train = data_result_train[idx[4]:idx[5]].sort_values(
        'REAL', ascending=False).reset_index(drop=True)
    d6_train = data_result_train[idx[5]:].sort_values(
        'REAL', ascending=False).reset_index(drop=True)


    min_test = data_result_test['REAL'].min()
    idx = list(data_result_test[data_result_test['REAL'] == min_test].index)# 0.10000000149011612].index)

    d1_test = data_result_test[idx[0]:].sort_values(
        'REAL', ascending=False).reset_index(drop=True)
    #d2_test = data_result_test[idx[1]:idx[2]].sort_values(
    #    'REAL', ascending=False).reset_index(drop=True)
    #d3_test = data_result_test[idx[2]:].sort_values(
    #    'REAL', ascending=False).reset_index(drop=True)

    dfs_train = [d1_train, d3_train, d4_train, d2_train, d5_train, d6_train]
    dfs_test = [d1_test]#, d2_test, d3_test]

    rejuste_index(dfs_train)
    rejuste_index(dfs_test)

    reajuste_predictions(dfs_train)
    reajuste_predictions(dfs_test)
    
    df_train_scored = pd.concat(dfs_train, axis=0)
    df_test_scored = pd.concat(dfs_test, axis=0)

    mae_train = mean_absolute_error(df_train_scored['REAL'], df_train_scored['PREDICTION'])
    mae_test = mean_absolute_error(df_test_scored['REAL'], df_test_scored['PREDICTION'])

    mape_train = mean_absolute_percentage_error(df_train_scored['REAL'], df_train_scored['PREDICTION'])
    mape_test = mean_absolute_percentage_error(df_test_scored['REAL'], df_test_scored['PREDICTION'])

    print(f"EPOCH: {ep} /n/n")

    print('-- RMSE --/n')
    print(f'mae train: {mae_train}')
    print(f'mae test: {mae_test} /n')

    print('-- RMSE PERC --/n')
    print(f'mape train: {mape_train}')
    print(f'mape test: {mape_test}')

    print("-"*30)
    
    title_name = "Rolamento1_6"
    compare_plot(d1_train, d2_train, d3_train, d4_train, d5_train, d6_train, d1_test, mape_test=mape_test, epochs=epochs, titile_test_grap_name=title_name) #, d2_test, d3_test)
    plt.savefig(f'images/{title_name}_resultados_epochs_{epochs}.png')