import os
import gzip
import re
import h5py
import numpy as np
from datetime import datetime

def import_gz_to_hdf5_datasets(gz_directory_path, hdf5_filepath):
    sub_directory_array = os.listdir(gz_directory_path+'/Data/')
    count = 0
    t = datetime.now()
    for filename in os.scandir(gz_directory_path+'/Data/'+sub_directory_array[0]):
        if filename.name.endswith(".gz"):
            fileidentifier = re.sub('\.gz', '', filename.name)
            with gzip.open(gz_directory_path+'/Data/'+sub_directory_array[0]+'/'+filename.name, 'rt') as f:
                data_array = np.array(
                    f.readline().rstrip().split(" ")).astype(np.float)
                for j in range(1, len(sub_directory_array)):
                    with gzip.open(gz_directory_path+'/Data/'+sub_directory_array[j]+'/'+filename.name, 'rt') as g:
                        sub_data_array = np.array(
                            g.readline().rstrip().split(" ")).astype(np.float)
                        data_array = np.append(data_array, sub_data_array)
                with gzip.open(gz_directory_path+'/Label/'+filename.name, 'rt') as h:
                    label_data_array = np.array(
                        h.readline().rstrip().split(" ")).astype(np.float).astype(np.int)
                _write_to_h5(hdf5_filepath, data_array,
                             label_data_array, fileidentifier)
            count += 1
            if((count<1000 and count%10 ==0) or (count%10000==0)):
                print(f'{count} files added in {datetime.now()-t} time')
        else:
            continue
			
def _write_to_h5(h5_filepath, values, label, identifier):
    with h5py.File(h5_filepath,'a') as hdf:
      group = hdf.create_group(identifier)
      group.create_dataset('data',data=values)
      group.create_dataset('label',data=label)
