"""Bearing 1"""

import pprint
import pandas as pd
import numpy as np
import os
import tqdm
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.utils import plot_model 
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from utils import PrepareData, compare_plot

def rejuste_index(lista_dfs):
    """Rejusta o index para o plot do RUL."""
    for index, df in enumerate(lista_dfs):
        if index == 0:
            df['index'] = df.index.tolist()
        else:
            shape_anterior = lista_dfs[index-1]['index'].tail(1).values[0]
            df['index'] = np.arange(shape_anterior, df.shape[0] + shape_anterior)
        df['PREDICTION'] = df['PREDICTION'].apply(lambda x: 0 if x<0 else x)

path = 'data_condition_1/'
data = os.listdir(path)

for i in data:
    globals()[f"{i}"] = pd.read_csv(path+i).drop('Unnamed: 0', axis=1)


# train
array1_1, rul1_1 = PrepareData.get_spectogram_array(data_Bearing1_1, 75, 'Horiz_accel', 'rul')
array1_2, rul1_2 = PrepareData.get_spectogram_array(data_Bearing1_2, 75, 'Horiz_accel', 'rul')
array1_5, rul1_5 = PrepareData.get_spectogram_array(data_Bearing1_5, 75, 'Horiz_accel', 'rul')
array1_6, rul1_6 = PrepareData.get_spectogram_array(data_Bearing1_6, 75, 'Horiz_accel', 'rul')

# test
array1_3, rul1_3 = PrepareData.get_spectogram_array(data_Bearing1_3, 75, 'Horiz_accel', 'rul')
array1_4, rul1_4 = PrepareData.get_spectogram_array(data_Bearing1_4, 75, 'Horiz_accel', 'rul')
array1_7, rul1_7 = PrepareData.get_spectogram_array(data_Bearing1_7, 75, 'Horiz_accel', 'rul')

# train
train = np.concatenate((array1_1, array1_2, array1_5, array1_6), axis=0)
label_train = np.concatenate((rul1_1, rul1_2, rul1_5, rul1_6), axis=0)

# test
test = np.concatenate((array1_3, array1_4, array1_7), axis=0)
label_test = np.concatenate((rul1_3, rul1_4, rul1_7), axis=0)

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

# model definition
model2 = keras.Sequential()

# layers
model2.add(keras.layers.Conv2D(
    filters=32,
    kernel_size=5,
    padding='same', 
    activation='relu',
    input_shape=train.shape[1:]))

model2.add(keras.layers.MaxPooling2D(
    pool_size=[2,2], 
    strides=2))

model2.add(keras.layers.Conv2D(
    filters=64,
    kernel_size=5,
    padding='same',
    activation='relu',
    input_shape=train.shape[1:]))

model2.add(keras.layers.MaxPooling2D(
    pool_size=[2,2],
    strides=2))

model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(256, activation='relu'))
model2.add(keras.layers.Dense(256, activation='relu'))
model2.add(keras.layers.Dense(1, activation='linear'))

# compile
model2.compile(optimizer='adam', loss='mse', metrics=['mse'])

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
epochs = 20
history = model2.fit(
    train, label_train, epochs=epochs, batch_size=50,
    validation_split=0.2, callbacks=[callback])

plot_model(model2, 'images/model.png', show_shapes=True)

plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.plot(range(1,epochs+1), history.history['mse'], label='train')
plt.plot(range(1,epochs+1), history.history['val_mse'], label='val')
plt.xticks(range(1,epochs+1))
plt.legend()
plt.tight_layout()
plt.savefig('images/mse_val_train.png')


predict_rul_test = model2.predict(test)
predict_rul_train = model2.predict(train)

predict_rul_test = [i[0] for i in predict_rul_test.tolist()]
predict_rul_train = [i[0] for i in predict_rul_train.tolist()]

data_result_train = pd.DataFrame([predict_rul_train, label_train]).T
data_result_test = pd.DataFrame([predict_rul_test, label_test]).T

coluns = ['PREDICTION', 'REAL']
data_result_train.columns = coluns
data_result_test.columns = coluns

idx = list(data_result_train[data_result_train['REAL'] == 10.0].index)

d1_train = data_result_train[idx[0]:idx[1]].sort_values('REAL', ascending=False).reset_index(drop=True)
d2_train = data_result_train[idx[1]:idx[2]].sort_values('REAL', ascending=False).reset_index(drop=True)
d3_train = data_result_train[idx[2]:idx[3]].sort_values('REAL', ascending=False).reset_index(drop=True)
d4_train = data_result_train[idx[3]:].sort_values('REAL', ascending=False).reset_index(drop=True)

idx = list(data_result_test[data_result_test['REAL'] == 10.0].index)

d1_test = data_result_test[idx[0]:idx[1]].sort_values('REAL', ascending=False).reset_index(drop=True)
d2_test = data_result_test[idx[1]:idx[2]].sort_values('REAL', ascending=False).reset_index(drop=True)
d3_test = data_result_test[idx[2]:].sort_values('REAL', ascending=False).reset_index(drop=True)

dfs_train = [d1_train, d3_train, d4_train, d2_train]
dfs_test = [d1_test, d2_test, d3_test]

rejuste_index(dfs_train)
rejuste_index(dfs_test)

compare_plot(d1_train, d2_train, d3_train, d4_train, d1_test, d2_test, d3_test)
plt.savefig(f'images/resultados_epochs_{epochs}.png')