import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compare_plot(
        d1_train, d2_train, d3_train, d4_train, d5_train, d6_train, d1_test, mape_test, epochs,
        alpha_pred=0.25, linewidth=2, titile_test_grap_name="test") -> any:
    #plt.figure(figsize=(15, 5), dpi=100)
    plt.figure(figsize=(11,6))

    plt.style.use('bmh')
    color_real =  "#06142E" # "#2B124C" 
    color_pred =   "#1B3358" # "#522B5B" 

    # TRAIN
    #plt.subplot(1, 2, 1)
    #dfs_train = [d1_train, d2_train, d3_train, d4_train, d5_train, d6_train]
    #len_ = len(dfs_train)
    #for index, data in enumerate(dfs_train):
    
    #    scaler = MinMaxScaler()
    #    x = data['index']
    
    #    y = data['REAL']  # scaler.fit_transform(
    #    # data['REAL'].values.reshape(-1, 1)).reshape(-1)
    #    y2 = data['PREDICTION']  # scaler.fit_transform(
    #    # data['PREDICTION'].values.reshape(-1, 1)).reshape(-1)
    
    #    plt.title('TRAIN')
    #    if index == len_ - 1:
    #        label_pred = 'PREDICTION'
    #        label_real = 'REAL'
    #        label_pred_trend = 'PREDICTION TREND'
    #    else:
    #        label_pred = None
    #        label_real = None
    #        label_pred_trend = None
    
    #    plt.plot(x, y2, color=color_pred,
    #                label=label_pred, alpha=alpha_pred)
    #    plt.plot(x, y, color=color_real,
    #             label=label_real, linewidth=linewidth)
    
    #    #z2 = np.polyfit(x, y2, 1)
    #    #p2 = np.poly1d(z2)
    #    #
    #    #plt.plot(x, p2(x), "r-", label=label_pred_trend, linewidth=linewidth)
    
    #plt.legend(frameon=True, facecolor='white', loc='upper left')
    
    ## TEST
    #plt.subplot(1, 2, 2)

    dfs_test = [d1_test]
    len_ = len(dfs_test)

    for index, data in enumerate(dfs_test):

        scaler = MinMaxScaler(feature_range=(0, 100))

        x = data['index']
        y = scaler.fit_transform(data['REAL'].values.reshape(-1, 1)).reshape(-1)
        y2 = scaler.transform(data['PREDICTION'].values.reshape(-1, 1)).reshape(-1)

        plt.title(titile_test_grap_name)
        plt.text(
            250, 
            10, 
            f"MAPE: {round(mape_test, 2)}\nEpochs: {epochs}", 
            fontsize=8, 
            alpha=0.5,
            fontstyle='italic',
            horizontalalignment='center',
            verticalalignment='center')

        if index == len_ - 1:
            label_pred = 'PREDICTION'
            label_real = 'REAL'
            label_pred_trend = 'PREDICTION TREND'
        else:
            label_pred = None
            label_real = None
            label_pred_trend = None

        plt.plot(x, y2, color=color_pred,
                    label=label_pred, alpha=alpha_pred)
        plt.plot(x, y, color=color_real,
                 label=label_real, linewidth=linewidth)

        #z2 = np.polyfit(x, y2, 1)
        #p2 = np.poly1d(z2)

        #plt.plot(x, p2(x), color="#f79646", label=label_pred_trend, linewidth=linewidth)
        
        #plt.ylim(bottom=0, top=101)
        plt.xlim(left=-20, right=2300)
        plt.tick_params(axis='x', labelsize=6)  # Ajusta o tamanho dos valores no eixo x
        plt.tick_params(axis='y', labelsize=6)
        plt.xlabel('Time (s)', fontname='Times New Roman', fontsize=15)
        plt.ylabel('RUL (%)', fontname='Times New Roman', fontsize=15)

    plt.legend(frameon=True, facecolor='white', loc='upper right')
