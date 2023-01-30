import os
from utils.prepare_data import PrepareData

prepare = PrepareData()

# https://www.kaggle.com/datasets/alanhabrony/ieee-phm-2012-data-challenge?select=ieee-phm-2012-data-challenge-dataset-master
path = 'data/ieee-phm-2012-data-challenge-dataset-master/'
train_set = 'Learning_set/'
test_set = 'Full_Test_Set/'

paths = [path+train_set, path+test_set]

for path in paths:
    data_raw = os.listdir(path)
    
    listas = []
    for i in data_raw:
        listas.append(path+i)
    listas.sort()
    
    for i in listas:
        lista = os.listdir(i)
        lista.sort()
        
        name = i.split('/')[-1]
        df_ = prepare.get_dataframe_acc(lista_inicial=lista, file=i)
    
        condition = name.split('_')[0][-1]
        folder = f"data_condition_{condition}"
    
        if folder not in os.listdir():
            os.mkdir(folder)
    
        df_.to_csv(f"{folder}/data_{name}")