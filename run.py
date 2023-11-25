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


epochs = 50
train, test, label_train, label_test = get_initial_input()
model2, history = generate_model(
    train=train, label_train=label_train, epochs=epochs)


# plt.title('Learning Curves')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.plot(range(1, epochs+1), history.history['mse'], label='train')
# plt.plot(range(1, epochs+1), history.history['val_mse'], label='val')
# plt.xticks(range(1, epochs+1))
# plt.legend()
# plt.tight_layout()
# plt.savefig('images/mse_val_train.png')

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

idx = list(
    data_result_train[data_result_train['REAL'] == 0.10000000149011612].index)

d1_train = data_result_train[idx[0]:idx[1]].sort_values(
    'REAL', ascending=False).reset_index(drop=True)
d2_train = data_result_train[idx[1]:idx[2]].sort_values(
    'REAL', ascending=False).reset_index(drop=True)
d3_train = data_result_train[idx[2]:idx[3]].sort_values(
    'REAL', ascending=False).reset_index(drop=True)
d4_train = data_result_train[idx[3]:].sort_values(
    'REAL', ascending=False).reset_index(drop=True)

idx = list(data_result_test[data_result_test['REAL']
           == 0.10000000149011612].index)

d1_test = data_result_test[idx[0]:idx[1]].sort_values(
    'REAL', ascending=False).reset_index(drop=True)
d2_test = data_result_test[idx[1]:idx[2]].sort_values(
    'REAL', ascending=False).reset_index(drop=True)
d3_test = data_result_test[idx[2]:].sort_values(
    'REAL', ascending=False).reset_index(drop=True)

dfs_train = [d1_train, d3_train, d4_train, d2_train]
dfs_test = [d1_test, d2_test, d3_test]

rejuste_index(dfs_train)
rejuste_index(dfs_test)

compare_plot(d1_train, d2_train, d3_train, d4_train, d1_test, d2_test, d3_test)
plt.savefig(f'images/resultados_epochs_{epochs}.png')

# TODO
# FAZER ESSE RUL DE FORMA PERCENETUAL VARIANDO ENTRE 0 E 1
df_train_scored = pd.concat(dfs_train, axis=0)
df_test_scored = pd.concat(dfs_test, axis=0)
# dd


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


rmse_train = rmse(df_train_scored['PREDICTION'], df_train_scored['REAL'])
rmse_test = rmse(df_test_scored['PREDICTION'], df_test_scored['REAL'])

print('-- RMSE --/n')
print(f'rmse train: {rmse_train}')
print(f'rmse test: {rmse_test}')
