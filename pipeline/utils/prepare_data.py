import os
from typing import Union, List
import pandas as pd
import tqdm
from scipy.signal import spectrogram
import numpy as np


class PrepareData:

    def __init__(
        self,
        path: str = ("../data/ieee-phm-2012-data-challenge-"
        "dataset-master/Full_Test_Set/"),
        columns: Union[str, List] = 'all') -> None:
        self.path = path
        self.__columns = columns
        columns_ = [
                    'Hour', 
                    'Minute', 
                    'Second', 
                    'mi_second', 
                    'Horiz_accel', 
                    'Vert_accel'
                ]

        if self.__columns == 'all':
            self.columns = columns_
        elif isinstance(self.__columns, list):
            unique_values = set(self.__columns) 
            if all([i in set(columns_) for i in unique_values]):
                self.columns = self.__columns
        else:
            raise ValueError(f"Columns must be passed as 'all' or as a"
                             f"list with elements presents in {columns_}")
            

    def get_dataframe_acc(self, lista_inicial: list, file: str) -> pd.DataFrame:
        """Prepare the dataframes for which bearing.
        
        Parameters
        ----------
        lista_inicial: list
            List of .csv's in which may starts with 'acc' and 'temp'
        file: str
            Example: 
            ../data/ieee-phm-2012-data-challenge-dataset-master/Full_Test_Set/Bearing1_3
        """

        lista = PrepareData.get_list_of_acc(lista_inicial)
        rul = 10*len(lista)

        df = pd.DataFrame(columns=self.columns)
        for i in tqdm.tqdm(lista, file.split('/')[-1]):
            try:
                df_aux = pd.read_csv(file+ '/' +i, header=None)
                df_aux.columns = self.columns

            except:
                df_aux = pd.read_csv(file+ '/' +i, header=None, sep=';')
                df_aux.columns = self.columns

            df_aux['rul'] = rul
            df = pd.concat([df,df_aux])
            rul -= 10

        return df


    @staticmethod
    def get_list_of_acc(lista):
        """Return a list of vibration data.
        
        Parameters
        ----------
        lista: list
            List of .csv's in which may starts with 'acc' and 'temp'
        """
        idx = PrepareData.get_inx_temp(lista)
        return lista[:idx]


    @staticmethod
    def get_inx_temp(lista: list):
        """Return index from the last vibration data.
        
        Parameters
        ----------
        lista: list
            List of .csv's in which may starts with 'acc' and 'temp'
        """
        for i in lista:
            if i.startswith('temp'):
                return lista.index(i)

    @staticmethod
    def get_spectogram_array(
        df: pd.DataFrame,
        split: int = 75,
        colum: str = 'Horiz_accel',
        rul_colum: str = 'rul'):
        """Return a list of spectogram arrays"""
    
        arrays = []
        ruls = []
        for rul, df_ in df[[colum, rul_colum]].groupby(rul_colum):
            # get the matrix of spectogram
            array = spectrogram(df_[colum], nperseg=split)[2].T
            arrays.append(array)
            ruls.append(rul)

        final_array = np.array(arrays, dtype='float32')
        ruls = np.array(ruls, dtype='float32')
        return final_array, ruls