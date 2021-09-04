import os
import gzip
import re
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

standard_category_list = ['plasmid', 'extra-plasmid', 'chromosome']
standard_feature_list = ['7mers', 'circular',
                         'fragments', 'inc-fac', 'orit', 'rrna']
featureNames = {'orit': ['id', 'OriT-identity', 'OriT-length', 'OriT-bitscore', 'OriT-count'],
                'rrna': ['id', 'rRNA-length', 'rRNA-bitscore', 'rRNA-count'],
                'inc-fac': ['id', 'IF-identity', 'IF-length', 'IF-bitscore', 'IF-count'],
                'circular': ['id', 'Cir-alignment_a_mean', 'Cir-alignment_b_mean', 'Cir-mismatches mean', 'Cir-count'],
                '7mers': ['7mer - '+str(i) for i in range(8192)]}
logfile = "log.txt"
featureCounts = {}
batchCounts = []


def create_hdf5_datasets(directory, hdf5_filepath, plasmid_classes_csv, chromosome_classes_csv):
    print('\n======= HDF5 dataset Creator ========\n')

    print('Cheking the Directory Structure....')
    _checkFileStructure(directory, hdf5_filepath,
                        plasmid_classes_csv, chromosome_classes_csv)
    print(
        f'Directory is in the standard format. {batchCounts[0]} batches of plasmids and {batchCounts[1]} batches of chromosomes found')

    logfile = os.path.dirname(hdf5_filepath) + "/log.txt"
    print(f'\nAny detected issue will be logged in {logfile} file')

    print('\nExtracting Plasmid Features....')
    plasmid_features_df = _read_features(directory + "/plasmid/Data")
    _logToFile(f'Plasmids\n{featureCounts}\n')
    print(f'\n{len(plasmid_features_df)} entries of {len(plasmid_features_df.columns)-1} features extracted')

    print('\nGenerating Plasmid Labels....')
    plasmid_labels_df = pd.read_csv(
        directory + '/plasmid/target.csv', names=['batch', 'id', 'seq_ID'])
    plasmid_labels_df = plasmid_labels_df.drop_duplicates()
    print(f'\n{len(plasmid_labels_df)} labels detected')

    differences_df = pd.merge(plasmid_features_df[['id']], plasmid_labels_df[['id']], on='id', suffixes=('_features', '_labels'),
                              how='outer', indicator='Exist')

    plasmid_features_df = plasmid_features_df.merge(
        differences_df, on='id', how='inner')
    feature_only_count = len(
        plasmid_features_df[plasmid_features_df['Exist'] == 'left_only'])
    print(f'{feature_only_count} feature entries does not have matching labels and omitted')
    plasmid_features_df = plasmid_features_df[plasmid_features_df['Exist'] == 'both']
    plasmid_features_df = plasmid_features_df.drop('Exist', axis=1)

    plasmid_labels_df = plasmid_labels_df.merge(
        differences_df, on='id', how='inner')
    label_only_count = len(
        plasmid_labels_df[plasmid_labels_df['Exist'] == 'right_only'])
    print(f'{label_only_count} label entries does not have matching entries and omitted')
    plasmid_labels_df = plasmid_labels_df[plasmid_labels_df['Exist'] == 'both']
    plasmid_labels_df = plasmid_labels_df.drop('Exist', axis=1)

    plasmid_classes_df = pd.read_csv(plasmid_classes_csv)
    plasmid_labels_df = plasmid_labels_df.merge(
        plasmid_classes_df[['Accession', 'Phylum']], how='left', left_on='seq_ID', right_on='Accession')
    plasmid_labels_df["label"] = plasmid_labels_df.groupby(
        ['Phylum'], sort=False).grouper.group_info[0]
    print(f'{len(plasmid_labels_df)} labels classified into {plasmid_labels_df.label.nunique()} plasmid classes')

    plasmid_classifed_df = pd.merge(
        plasmid_labels_df, plasmid_features_df, on='id', how='outer')

    print('\nExtracting Chromosome Features....')
    chromosome_features_df = _read_features(directory + "/chromosome/Data")
    _logToFile(f'Chromosomes\n{featureCounts}\n')
    print(f'\n{len(chromosome_features_df)} entries of {len(chromosome_features_df.columns)-1} features extracted')

    print('\nGenerating Chromosome Labels....')
    chromosome_labels_df = pd.read_csv(
        directory + '/chromosome/target.csv', names=['batch', 'id', 'seq_ID'])
    chromosome_labels_df = chromosome_labels_df.drop_duplicates()
    print(f'\n{len(chromosome_labels_df)} labels detected')

    differences_df = pd.merge(chromosome_features_df[['id']], chromosome_labels_df[['id']], on='id', suffixes=('_features', '_labels'),
                              how='outer', indicator='Exist')

    chromosome_features_df = chromosome_features_df.merge(
        differences_df, on='id', how='inner')
    feature_only_count = len(
        chromosome_features_df[chromosome_features_df['Exist'] == 'left_only'])
    print(f'{feature_only_count} feature entries does not have matching labels and omitted')
    chromosome_features_df = chromosome_features_df[chromosome_features_df['Exist'] == 'both']
    chromosome_features_df = chromosome_features_df.drop('Exist', axis=1)

    chromosome_labels_df = chromosome_labels_df.merge(
        differences_df, on='id', how='inner')
    label_only_count = len(
        chromosome_labels_df[chromosome_labels_df['Exist'] == 'right_only'])
    print(f'{label_only_count} label entries does not have matching entries and omitted')
    chromosome_labels_df = chromosome_labels_df[chromosome_labels_df['Exist'] == 'both']
    chromosome_labels_df = chromosome_labels_df.drop('Exist', axis=1)

    chromosome_classes_df = pd.read_csv(chromosome_classes_csv)
    chromosome_labels_df = chromosome_labels_df.merge(
        chromosome_classes_df[['Assembly_Accession', 'Phylum']], how='left', left_on='seq_ID', right_on='Assembly_Accession')
    chromosome_labels_df["label"] = chromosome_labels_df.groupby(
        ['Phylum'], sort=False).grouper.group_info[0]
    print(f'{len(chromosome_labels_df)} labels classified into {chromosome_labels_df.label.nunique()} chromosome classes')

    chromosome_classifed_df = pd.merge(
        chromosome_labels_df, chromosome_features_df, on='id', how='outer')

    # print(plasmid_features_df.head())
    # print(plasmid_labels_df.head())
    # print(plasmid_features_df.shape)
    # print(plasmid_labels_df.shape)

    chromosome_features_df['id'] = chromosome_features_df['id'].astype(str)+'_c'
    chromosome_labels_df['id'] = chromosome_labels_df['id'].astype(str)+'_c'

    print('\nWriting to the HDF5 file....')

    with h5py.File(hdf5_filepath, 'a') as hdf:
        print('\nWriting Plasmid Data....')
        plasmid_features_df.apply(
            lambda row: _write_data_to_h5(hdf, row), axis=1)
        print('Writing Plasmid Labels....')
        plasmid_labels_df.apply(
            lambda row: _write_label_to_h5(hdf, row), axis=1)
        print('Writing Chromosome Data....')
        chromosome_features_df.apply(
            lambda row: _write_data_to_h5(hdf, row), axis=1)
        print('Writing Chromosome Labels....')
        chromosome_labels_df.apply(
            lambda row: _write_label_to_h5(hdf, row), axis=1)

    print('\nHDF5 file completed')


def _write_data_to_h5(h5_file, row):
    data_id = row['id']
    data_array = row.drop('id').to_numpy(dtype=np.float)
    group = h5_file.create_group(str(data_id))
    group.create_dataset('data', data=data_array)


def _write_label_to_h5(h5_file, row):
    label_id = row['id']
    try:
        group = h5_file[str(label_id)]
        label_array = np.array([row['label']], dtype=np.int)
        group.create_dataset('label', data=label_array)
    except:
        _logToFile(
            f'label for {label_id} omitted because no data group was available\n')
        return


def _checkFileStructure(directory, hdf5_filepath, plasmid_classes_csv, chromosome_classes_csv):
    if (os.path.exists(hdf5_filepath)):
        print(f'{hdf5_filepath} already available. appending to it.')
    if (not (os.path.exists(plasmid_classes_csv) and os.path.exists(chromosome_classes_csv))):
        raise RuntimeError('required class CSVs not found')
    if(not os.path.isdir(directory)):
        raise RuntimeError(f'The given path is not a directory')
    sub_directory_array = os.listdir(directory)
    if(len(sub_directory_array) != len(standard_category_list)):
        raise RuntimeError(
            f'The directory should contain {len(standard_category_list)} categories')
    for category_name in sub_directory_array:
        if (category_name == "extra-plasmid"):
            continue
        if(not os.path.exists(directory+"/"+category_name+"/target.csv")):
            raise RuntimeError(f'Can not place targets.csv in {category_name}')
        if(category_name not in standard_category_list):
            raise RuntimeError(
                f'The categories should be {",".join(standard_category_list)}')
        feature_array = os.listdir(directory+"/"+category_name+"/Data")
        if(len(feature_array) != len(standard_feature_list)):
            raise RuntimeError(
                f'The directory should contain {len(standard_feature_list)} features for {category_name}')
        batch_count = 0
        for feature_name in feature_array:
            if(feature_name not in standard_feature_list):
                raise RuntimeError(
                    f'The features should be {",".join(standard_feature_list)} for {category_name}')
            if (feature_name == 'fragments'):
                continue
            feature_batch_count = len(os.listdir(
                directory+"/"+category_name+"/Data/"+feature_name))
            if(feature_batch_count == 0):
                raise RuntimeError(
                    f'No files in {feature_name} of {category_name}')
            elif(batch_count == 0):
                batch_count = feature_batch_count
            elif (feature_batch_count != batch_count):
                raise RuntimeError(
                    f'No of batches in {feature_name} of {category_name} does not match with previous batch count')
        batchCounts.append(batch_count)

def _read_features(path):
    featurefiles = os.listdir(path)
    biomer_dfs = []
    print('\nReading 7mers....')
    feature_df = _read_feature_files(
        path + '/7mers', ['7mer-' + str(i) for i in range(8192)])
    feature_df['id'] = range(len(feature_df))
    featureCounts['7mers'] = len(feature_df)
    print('\nReading other features....')
    for ff in featurefiles:
        if (ff == 'fragments' or ff == '7mers'):
            continue
        print(ff)
        single_f_df = _read_feature_files(path + '/' + ff, featureNames[ff])
        featureCounts[ff] = len(single_f_df)
        feature_df = feature_df.merge(single_f_df, how='inner', on='id')
    return feature_df


def _read_feature_files(path, feature_names):
    files = os.listdir(path)
    files.sort()
    fileArrays = []
    for file in tqdm(files):
        fileArrays.append(np.genfromtxt(path+"/"+file, dtype=np.int32))
    featureArray = np.concatenate(fileArrays)
    feature_df = pd.DataFrame(featureArray, columns=feature_names)
    return feature_df


def _logToFile(msg):
    with open(logfile, 'a') as log:
        log.write(msg)
