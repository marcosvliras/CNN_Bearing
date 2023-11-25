import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compare_plot(
        d1_train, d2_train, d3_train, d4_train, d1_test, d2_test, d3_test,
        alpha_pred=0.05, linewidth=2) -> any:
    plt.figure(figsize=(15, 5), dpi=100)

    # TRAIN
    plt.subplot(1, 2, 1)
    dfs_train = [d1_train, d2_train, d3_train, d4_train]
    len_ = len(dfs_train)
    for index, data in enumerate(dfs_train):

        scaler = MinMaxScaler()
        x = data['index']

        y = data['REAL']  # scaler.fit_transform(
        # data['REAL'].values.reshape(-1, 1)).reshape(-1)
        y2 = data['PREDICTION']  # scaler.fit_transform(
        # data['PREDICTION'].values.reshape(-1, 1)).reshape(-1)

        plt.title('TRAIN')
        if index == len_ - 1:
            label_pred = 'PREDICTION'
            label_real = 'REAL'
            label_pred_trend = 'PREDICTION TREND'
        else:
            label_pred = None
            label_real = None
            label_pred_trend = None

        plt.scatter(x, y2, color='darkgreen',
                    label=label_pred, alpha=alpha_pred)
        plt.plot(x, y, color='lightgreen',
                 label=label_real, linewidth=linewidth)

        z2 = np.polyfit(x, y2, 1)
        p2 = np.poly1d(z2)

        plt.plot(x, p2(x), "r-", label=label_pred_trend, linewidth=linewidth)

    plt.legend(frameon=True, facecolor='white', loc='upper left')

    # TEST
    plt.subplot(1, 2, 2)

    dfs_test = [d1_test, d2_test, d3_test]
    len_ = len(dfs_test)

    for index, data in enumerate(dfs_test):

        scaler = MinMaxScaler()

        x = data['index']
        y = data['REAL']  # scaler.fit_transform(
        # data['REAL'].values.reshape(-1, 1)).reshape(-1)
        y2 = data['PREDICTION']  # scaler.fit_transform(
        # data['PREDICTION'].values.reshape(-1, 1)).reshape(-1)

        plt.title('TEST')

        if index == len_ - 1:
            label_pred = 'PREDICTION'
            label_real = 'REAL'
            label_pred_trend = 'PREDICTION TREND'
        else:
            label_pred = None
            label_real = None
            label_pred_trend = None

        plt.scatter(x, y2, color='darkgreen',
                    label=label_pred, alpha=alpha_pred)
        plt.plot(x, y, color='lightgreen',
                 label=label_real, linewidth=linewidth)

        z2 = np.polyfit(x, y2, 1)
        p2 = np.poly1d(z2)

        plt.plot(x, p2(x), "r-", label=label_pred_trend, linewidth=linewidth)

    plt.legend(frameon=True, facecolor='white', loc='upper right')
