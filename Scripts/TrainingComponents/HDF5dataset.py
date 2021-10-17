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
    def __init__(self, file_path, load_data, data_cache_size=3, plas_chro_labeling=True, label_threshold = None):
        super().__init__()
        self.data_info = []
        self.label_info = []
        self.data_cache = {}
        self.h5_filepath = file_path
        self.plas_chro_labeling = plas_chro_labeling
        self.label_threshold = label_threshold
        self.data_cache_size = data_cache_size
        self._add_data_infos(load_data)
        self.df_label_info = pd.DataFrame(self.label_info)

    def __getitem__(self, index):
        x = self.get_data("data", index)
        x = torch.from_numpy(x)

        y = self.get_data("label", index)
        y = torch.tensor(y[0])
        if (self.plas_chro_labeling):
            if (y.item() > self.label_threshold):
                y = torch.tensor(1)
            else:
                y = torch.tensor(0)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, load_data):
        with h5py.File(self.h5_filepath) as h5_file:
            for gname, group in tqdm(h5_file.items()):
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds[()], gname)

                    if (dname == 'label'):
                        val = ds[0]
                        self.data_info.append({'file_id': gname, 'value': val, 'type': dname, 'shape': ds.shape, 'cache_idx': idx})
                        self.label_info.append(
                            {'file_id': gname, 'value': val})
                    else:
                        self.data_info.append({'file_id':gname, 'value':'DATA', 'type': dname, 'shape': ds.shape, 'cache_idx': idx})

    def _load_data(self, fid):

        with h5py.File(self.h5_filepath) as h5_file:
            group = h5_file[fid]
            for dname, ds in group.items():
                idx = self._add_to_cache(ds[()], fid)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i, v in enumerate(
                    self.data_info) if v['file_id'] == fid)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(fid)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_id': di['file_id'],'value':di['value'], 'type': di['type'], 'shape': di['shape'],
                               'cache_idx': -1} if di['file_id'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, data_id):

        if data_id not in self.data_cache:
            self.data_cache[data_id] = [data]
        else:
            self.data_cache[data_id].append(data)
        return len(self.data_cache[data_id]) - 1

    def get_data_infos(self, type):

        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):

        fid = self.get_data_infos(type)[i]['file_id']
        if fid not in self.data_cache:
            self._load_data(fid)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fid][cache_idx]

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
