import h5py
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils import data
import re
from datetime import datetime


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, isTraining,data_cache_size=3, plas_chro_labeling=True, label_threshold=20):
        super().__init__()
        if (isTraining):
            self.data_info = pd.read_csv(
                file_path+'/h5_data_infos/training_data_info.csv')
            self.label_info = pd.read_csv(
                file_path + '/h5_data_infos/training_label_info.csv')
        else:
            self.data_info = pd.read_csv(
                file_path+'/h5_data_infos/testing_data_info.csv')
            self.label_info = pd.read_csv(
                file_path + '/h5_data_infos/testing_label_info.csv')
            
        self.data_cache = {}
        self.plas_chro_labeling = plas_chro_labeling
        self.label_threshold = label_threshold
        self.data_cache_size = data_cache_size

    def __getitem__(self, index):
        x, fidx = self.get_data("data", index)
        x = torch.from_numpy(x)

        y, fidy = self.get_data("label", index)
        y = torch.tensor(y[0])
        if (self.plas_chro_labeling):
            if (y.item() > self.label_threshold):
                y = torch.tensor(0)
            else:
                y = torch.tensor(1)

        fid = "wrong"
        if(fidx == fidy):
            fid = fidx

        return (x, y, fid)

    def __len__(self):
        return len(self.get_data_infos('data'))

    
    def _load_data(self, fid, fpath, findex):

        with h5py.File(fpath) as h5_file:
            group = h5_file[fid]
            for dname, ds in group.items():
                idx = self._add_to_cache(ds[()], fid)

                self.data_info.at[findex, 'cache_idx'] = idx
                self.data_info.at[findex+1, 'cache_idx'] = idx+1


        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_key = list(self.data_cache.keys())[0]
            self.data_cache.pop(removal_key)
            # remove invalid cache_idx
            self.data_info.loc[self.data_info['file_id'] == removal_key, 'cache_idx'] = -1

    def _add_to_cache(self, data, data_id):

        if data_id not in self.data_cache:
            self.data_cache[data_id] = [data]
        else:
            self.data_cache[data_id].append(data)
        return 0

    def get_data_infos(self, type):

        data_info_type = self.data_info.loc[self.data_info['type'] == type]
        return data_info_type

    def get_data(self, type, i):

        temp = self.get_data_infos(type).iloc[[i]]
        fid = str(temp['file_id'].item())
        fpath = str(temp['file'].item())
        findex = int(temp.index.item())

        if fid not in self.data_cache:
            self._load_data(fid, fpath, findex)

        # get new cache_idx assigned by _load_data_info
        cache_idx = int(self.get_data_infos(type).iloc[[i]]['cache_idx'].item())
        return self.data_cache[fid][cache_idx], fid

    def get_label_values(self, indices):
        t = datetime.now()
        lbls = torch.zeros(len(indices), dtype=torch.int)
        print('Retriving labels for training set......')
        for i in tqdm(range(len(indices))):
            val = int(self.df_label_info.at[indices[i].item(), 'value'])
            if (self.plas_chro_labeling):
                if (val > self.label_threshold):
                    val = 1
                else:
                    val = 0
            lbls[i] = val
        return lbls
