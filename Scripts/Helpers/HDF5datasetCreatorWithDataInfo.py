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
logfile = None
featureCounts = {}
batchFiles = []


def create_hdf5_datasets(directory, plasmid_classes_csv, chromosome_classes_csv, hdf5_path=None, test_fraction=0.2, batches_per_iteration=1, plasmid_batch_limit=None, chromosome_batch_limit=None):
    global logfile
    plasmid_class_count = 0
    chromosome_class_count = 0
    plas_limit = plasmid_batch_limit
    chrom_limit = chromosome_batch_limit
    curr_plas_batches = 0
    curr_chrom_batches = 0
    training_data_info = []
    testing_data_info = []
    training_label_info = []
    testing_label_info = []

    print('\n======= HDF5 dataset Creator ========\n')
    if (hdf5_path == None):
        hdf5_path = directory + '/'
    else:
        hdf5_path = hdf5_path + '/'
    training_hdf5_path = hdf5_path+'training.h5'
    testing_hdf5_path = hdf5_path+'testing.h5'

    logfile = hdf5_path + "log.txt"
    print(f'Any detected issue will be logged in {logfile} file')

    print('Cheking the Directory Structure....')
    _checkFileStructure(directory, hdf5_path,
                        plasmid_classes_csv, chromosome_classes_csv)
    _logToFilewithPrint('Directory is in the standard format')
    if(plas_limit == None):
        plas_limit = len(batchFiles[0])
    if (chrom_limit == None):
        chrom_limit = len(batchFiles[1])
    _logToFilewithPrint(
        f'{len(batchFiles[0])} batches of plasmids and {len(batchFiles[1])} batches of chromosomes found')
    if(plas_limit != len(batchFiles[0]) or chrom_limit != len(batchFiles[1])):
        _logToFilewithPrint(
            f'{plas_limit} batches of plasmids and {chrom_limit} batches of chromosomes will be used')

    print('Surveying Plasmid Info....')
    plasmid_labels_df = pd.read_csv(
        directory + '/plasmid/target.csv', names=['batch', 'id', 'seq_ID'])
    plasmid_labels_df = plasmid_labels_df.drop_duplicates()
    if (plas_limit == len(batchFiles[0])):
        plasmid_labels_df = plasmid_labels_df.head(plas_limit*10000)
    _logToFilewithPrint(f'{len(plasmid_labels_df)} plasmid labels detected')

    plasmid_classes_df = pd.read_csv(plasmid_classes_csv)
    plasmid_labels_df = plasmid_labels_df.merge(
        plasmid_classes_df[['Accession', 'Phylum']], how='left', left_on='seq_ID', right_on='Accession')
    plasmid_labels_df["label"] = plasmid_labels_df.groupby(
        ['Phylum'], sort=False).grouper.group_info[0]

    plasmid_class_count = plasmid_labels_df.label.nunique()
    _logToFilewithPrint(
        f'{len(plasmid_labels_df)} labels classified into {plasmid_labels_df.label.nunique()} classes')

    temp_df = plasmid_labels_df.groupby(
        plasmid_labels_df['label']).sample(frac=test_fraction)
    temp_df['type'] = 'Test'
    plasmid_labels_df = plasmid_labels_df.merge(temp_df, how="outer", on=[
                                                'batch', 'id', 'seq_ID', 'Accession', 'Phylum', 'label'])
    plasmid_labels_df['type'] = plasmid_labels_df['type'].fillna("Train")

    temp_df1 = plasmid_labels_df.sample(frac=0.3333)
    temp_df1['group'] = 'G1'
    plasmid_labels_df = plasmid_labels_df.merge(temp_df1, how="left", on=[
                                                'batch', 'id', 'seq_ID', 'Accession', 'Phylum', 'label', 'type'])
    temp_df2 = plasmid_labels_df[plasmid_labels_df['group'].isna()].sample(
        frac=0.5)
    temp_df2['group'] = 'G2'
    plasmid_labels_df = plasmid_labels_df.merge(temp_df2, how="left", on=[
        'batch', 'id', 'seq_ID', 'Accession', 'Phylum', 'label', 'type'])
    plasmid_labels_df['group'] = plasmid_labels_df['group_x'].fillna(
        plasmid_labels_df['group_y']).fillna('G3')
    plasmid_labels_df.drop(
        ['group_x', 'group_y', 'Accession'], inplace=True, axis=1)

    print(plasmid_labels_df['label'].value_counts())
    print(plasmid_labels_df['group'].value_counts())
    plasmid_labels_df.to_csv(hdf5_path+'final_plasmid_labels.csv')

    print('Surveying Chromosome Info....')
    chromosome_labels_df = pd.read_csv(
        directory + '/chromosome/target.csv', names=['batch', 'id', 'seq_ID', 'dom_seq_ID'])
    chromosome_labels_df = chromosome_labels_df.drop_duplicates()
    if (chrom_limit == len(batchFiles[1])):
        chromosome_labels_df = chromosome_labels_df.head(chrom_limit*10000)
    _logToFilewithPrint(
        f'{len(chromosome_labels_df)} chromosome labels detected')

    chromosome_classes_df = pd.read_csv(chromosome_classes_csv)
    chromosome_labels_df = chromosome_labels_df.merge(
        chromosome_classes_df[['Assembly_Accession', 'Phylum']], how='left', left_on='dom_seq_ID', right_on='Assembly_Accession')
    chromosome_labels_df["label"] = chromosome_labels_df.groupby(
        ['Phylum'], sort=False).grouper.group_info[0]

    chromosome_labels_df["label"] = chromosome_labels_df["label"] + \
        (plasmid_class_count + 1)
    chromosome_class_count = chromosome_labels_df.label.nunique()
    _logToFilewithPrint(
        f'{len(chromosome_labels_df)} labels classified into {chromosome_labels_df.label.nunique()} classes')

    temp_df = chromosome_labels_df.groupby(
        chromosome_labels_df['label']).sample(frac=test_fraction)
    temp_df['type'] = 'Test'
    chromosome_labels_df = chromosome_labels_df.merge(
        temp_df, how="outer", on=['batch', 'id', 'seq_ID', 'Assembly_Accession', 'Phylum', 'label', 'dom_seq_ID'])
    chromosome_labels_df['type'] = chromosome_labels_df['type'].fillna("Train")


    temp_df1 = chromosome_labels_df.sample(frac=0.333333)
    temp_df1['group'] = 'G1'
    chromosome_labels_df = chromosome_labels_df.merge(temp_df1, how="left", on=[
        'batch', 'id', 'seq_ID', 'dom_seq_ID', 'Assembly_Accession', 'Phylum', 'label', 'type'])
    temp_df2 = chromosome_labels_df[chromosome_labels_df['group'].isna()].sample(
        frac=0.5)
    temp_df2['group'] = 'G2'
    chromosome_labels_df = chromosome_labels_df.merge(temp_df2, how="left", on=[
        'batch', 'id', 'seq_ID', 'dom_seq_ID', 'Assembly_Accession', 'Phylum', 'label', 'type'])
    chromosome_labels_df['group'] = chromosome_labels_df['group_x'].fillna(
        chromosome_labels_df['group_y']).fillna('G3')
    chromosome_labels_df.drop(
        ['group_x', 'group_y', 'Assembly_Accession'], inplace=True, axis=1)

    print(chromosome_labels_df['label'].value_counts())
    print(chromosome_labels_df['group'].value_counts())
    chromosome_labels_df.to_csv(hdf5_path + 'final_chromosome_labels.csv')

    for i in range(0, plas_limit, batches_per_iteration):
        _logToFilewithPrint(
            f'\n----ITERATION {int((i+batches_per_iteration)/batches_per_iteration)}----')
        print('\nExtracting Features....')
        if (curr_plas_batches + batches_per_iteration > plas_limit):
            iteration_batches = [m for m in range(
                i, i + (plas_limit - curr_plas_batches))]
        else:
            iteration_batches = [m for m in range(i, i+batches_per_iteration)]
        plasmid_features_df = _read_features(
            directory + "/plasmid/Data", iteration_batches)
        _logToFilewithPrint(
            f'{len(plasmid_features_df)} entries of {len(plasmid_features_df.columns)-1} features extracted')

        differences_df = pd.merge(plasmid_features_df[['id']], plasmid_labels_df[['id']], on='id', suffixes=('_features', '_labels'),
                                  how='outer', indicator='Exist')

        plasmid_features_df = plasmid_features_df.merge(
            differences_df, on='id', how='inner')
        feature_only_count = len(
            plasmid_features_df[plasmid_features_df['Exist'] == 'left_only'])
        _logToFilewithPrint(
            f'{feature_only_count} feature entries does not have matching labels and omitted')
        plasmid_features_df = plasmid_features_df[plasmid_features_df['Exist'] == 'both']
        plasmid_features_df = plasmid_features_df.drop('Exist', axis=1)

        plasmid_labels_duplicate_df = plasmid_labels_df.merge(
            differences_df, on='id', how='inner')

        label_only_count = len(
            plasmid_labels_duplicate_df[plasmid_labels_duplicate_df['Exist'] == 'right_only'])
        _logToFilewithPrint(
            f'{label_only_count} label entries does not have matching entries and omitted')
        plasmid_labels_duplicate_df = plasmid_labels_duplicate_df[
            plasmid_labels_duplicate_df['Exist'] == 'both']
        plasmid_labels_duplicate_df = plasmid_labels_duplicate_df.drop(
            'Exist', axis=1)

        plasmid_classifed_df = pd.merge(
            plasmid_labels_duplicate_df, plasmid_features_df, on='id', how='outer')

        plasmid_features_df['id'] = plasmid_features_df['id'].astype(
            str)+'_p'
        plasmid_labels_duplicate_df['id'] = plasmid_labels_duplicate_df['id'].astype(
            str) + '_p'

        print(plasmid_features_df.astype(bool).sum(axis=0))

        plasmid_features_df = plasmid_features_df.merge(
            plasmid_labels_duplicate_df[['id', 'type']], on='id', how='left')
        plasmid_features_train_df = plasmid_features_df[
            plasmid_features_df['type'] == 'Train']
        plasmid_features_test_df = plasmid_features_df[plasmid_features_df['type'] == 'Test']
        plasmid_labels_duplicate_train_df = plasmid_labels_duplicate_df[
            plasmid_labels_duplicate_df['type'] == 'Train']
        plasmid_labels_duplicate_test_df = plasmid_labels_duplicate_df[
            plasmid_labels_duplicate_df['type'] == 'Test']

        _logToFilewithPrint(
            f'Training plasmid count:{len(plasmid_features_train_df)}')
        _logToFilewithPrint(
            f'Testing plasmid count:{len(plasmid_features_test_df)}')

        with h5py.File(training_hdf5_path, 'a') as hdf:
            _logToFilewithPrint(f'Training.h5 created at {training_hdf5_path}')
            print('Writing Training Data....')
            plasmid_features_train_df.apply(
                lambda row: _write_data_to_h5(hdf, row, training_hdf5_path, training_data_info), axis=1)
            print('Writing Training Labels....')
            plasmid_labels_duplicate_train_df.apply(
                lambda row: _write_label_to_h5(hdf, row, training_hdf5_path, training_data_info, training_label_info), axis=1)

        with h5py.File(testing_hdf5_path, 'a') as hdf:
            _logToFilewithPrint(f'Testing.h5 created at {testing_hdf5_path}')
            print('Writing Testing Data....')
            plasmid_features_test_df.apply(
                lambda row: _write_data_to_h5(hdf, row, testing_hdf5_path, testing_data_info), axis=1)
            print('Writing Testing Labels....')
            plasmid_labels_duplicate_test_df.apply(
                lambda row: _write_label_to_h5(hdf, row, testing_hdf5_path, testing_data_info, testing_label_info), axis=1)

        curr_plas_batches += len(iteration_batches)
        _logToFilewithPrint(
            f'\n{curr_plas_batches}/{plas_limit} PLASMID BATCHES COMPLETED')

    for i in range(0, chrom_limit, batches_per_iteration):
        _logToFilewithPrint(
            f'\n----ITERATION {int((i+batches_per_iteration)/batches_per_iteration)}----')
        print('\nExtracting Features....')
        if (curr_chrom_batches + batches_per_iteration > chrom_limit):
            iteration_batches = [m for m in range(
                i, i + (chrom_limit - curr_chrom_batches))]
        else:
            iteration_batches = [m for m in range(i, i+batches_per_iteration)]
        chromosome_features_df = _read_features(
            directory + "/chromosome/Data", iteration_batches)
        _logToFilewithPrint(
            f'{len(chromosome_features_df)} entries of {len(chromosome_features_df.columns)-1} features extracted')

        differences_df = pd.merge(chromosome_features_df[['id']], chromosome_labels_df[['id']], on='id', suffixes=('_features', '_labels'),
                                  how='outer', indicator='Exist')

        chromosome_features_df = chromosome_features_df.merge(
            differences_df, on='id', how='inner')
        feature_only_count = len(
            chromosome_features_df[chromosome_features_df['Exist'] == 'left_only'])
        _logToFilewithPrint(
            f'{feature_only_count} feature entries does not have matching labels and omitted')
        chromosome_features_df = chromosome_features_df[chromosome_features_df['Exist'] == 'both']
        chromosome_features_df = chromosome_features_df.drop('Exist', axis=1)

        chromosome_labels_duplicate_df = chromosome_labels_df.merge(
            differences_df, on='id', how='inner')
        label_only_count = len(
            chromosome_labels_duplicate_df[chromosome_labels_duplicate_df['Exist'] == 'right_only'])
        _logToFilewithPrint(
            f'{label_only_count} label entries does not have matching entries and omitted')
        chromosome_labels_duplicate_df = chromosome_labels_duplicate_df[
            chromosome_labels_duplicate_df['Exist'] == 'both']
        chromosome_labels_duplicate_df = chromosome_labels_duplicate_df.drop(
            'Exist', axis=1)

        chromosome_classifed_df = pd.merge(
            chromosome_labels_duplicate_df, chromosome_features_df, on='id', how='outer')

        chromosome_features_df['id'] = chromosome_features_df['id'].astype(
            str)+'_c'
        chromosome_labels_duplicate_df['id'] = chromosome_labels_duplicate_df['id'].astype(
            str) + '_c'

        print(chromosome_features_df.astype(bool).sum(axis=0))

        chromosome_features_df = chromosome_features_df.merge(
            chromosome_labels_duplicate_df[['id', 'type']], on='id', how='left')
        chromosome_features_train_df = chromosome_features_df[
            chromosome_features_df['type'] == 'Train']
        chromosome_features_test_df = chromosome_features_df[chromosome_features_df['type'] == 'Test']
        chromosome_labels_duplicate_train_df = chromosome_labels_duplicate_df[
            chromosome_labels_duplicate_df['type'] == 'Train']
        chromosome_labels_duplicate_test_df = chromosome_labels_duplicate_df[
            chromosome_labels_duplicate_df['type'] == 'Test']

        _logToFilewithPrint(
            f'Training chromosome count:{len(chromosome_features_train_df)}')
        _logToFilewithPrint(
            f'Testing chromosome count:{len(chromosome_features_test_df)}')

        with h5py.File(training_hdf5_path, 'a') as hdf:
            print('Writing Training Data....')
            chromosome_features_train_df.apply(
                lambda row: _write_data_to_h5(hdf, row, training_hdf5_path, training_data_info), axis=1)
            print('Writing Training Labels....')
            chromosome_labels_duplicate_train_df.apply(
                lambda row: _write_label_to_h5(hdf, row, training_hdf5_path, training_data_info, training_label_info), axis=1)

        with h5py.File(testing_hdf5_path, 'a') as hdf:
            print('Writing Testing Data....')
            chromosome_features_test_df.apply(
                lambda row: _write_data_to_h5(hdf, row, testing_hdf5_path, testing_data_info), axis=1)
            print('Writing Testing Labels....')
            chromosome_labels_duplicate_test_df.apply(
                lambda row: _write_label_to_h5(hdf, row, testing_hdf5_path, testing_data_info, testing_label_info), axis=1)

        curr_chrom_batches += len(iteration_batches)
        _logToFilewithPrint(
            f'\n{curr_chrom_batches}/{chrom_limit} CHROMOSOME BATCHES COMPLETED')

    os.mkdir(hdf5_path+'h5_data_infos')
    pd.DataFrame(training_data_info, columns=[
                 'file', 'file_id', 'value', 'type', 'shape', 'cache_idx']).to_csv(hdf5_path+'h5_data_infos/training_data_info.csv', index=False)
    pd.DataFrame(training_label_info, columns=[
                 'file', 'file_id', 'value']).to_csv(hdf5_path+'h5_data_infos/training_label_info.csv', index=False)
    pd.DataFrame(testing_data_info, columns=[
                 'file', 'file_id', 'value', 'type', 'shape', 'cache_idx']).to_csv(hdf5_path+'h5_data_infos/testing_data_info.csv', index=False)
    pd.DataFrame(testing_label_info, columns=[
        'file', 'file_id', 'value']).to_csv(hdf5_path+'h5_data_infos/testing_label_info.csv', index=False)


def _write_data_to_h5(h5_file, row, filepath, data_info):
    data_id = row['id']
    data_array = row.drop('id').drop('type').to_numpy(dtype=np.float64)
    group = h5_file.create_group(str(data_id))
    group.create_dataset('data', data=data_array)
    data_info.append(
        {'file': filepath, 'file_id': str(data_id), 'value': 'DATA', 'type': 'data', 'shape': data_array.shape, 'cache_idx': -1})


def _write_label_to_h5(h5_file, row, filepath, data_info, label_info):
    label_id = row['id']
    try:
        group = h5_file[str(label_id)]
        label_array = np.array([row['label']], dtype=np.int32)
        group.create_dataset('label', data=label_array)
        data_info.append(
            {'file': filepath, 'file_id': str(label_id), 'value': label_array[0], 'type': 'label', 'shape': label_array.shape, 'cache_idx': -1})
        label_info.append(
            {'file': filepath, 'file_id': str(label_id), 'value': label_array[0]})
    except:
        _logToFile(
            f'label for {label_id} omitted because no data group was available\n')
        return


def _checkFileStructure(directory, hdf5_path, plasmid_classes_csv, chromosome_classes_csv):
    if (os.path.exists(hdf5_path+'training.h5') or os.path.exists(hdf5_path+'testing.h5')):
        print(f'{hdf5_path} already available. appending to it.')
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
        batch_file_array = []
        for feature_name in feature_array:
            if(feature_name not in standard_feature_list):
                raise RuntimeError(
                    f'The features should be {",".join(standard_feature_list)} for {category_name}')
            if (feature_name == 'fragments'):
                continue
            feature_batch_files = os.listdir(
                directory+"/"+category_name+"/Data/"+feature_name)
            feature_batch_count = len(feature_batch_files)
            if(feature_batch_count == 0):
                raise RuntimeError(
                    f'No files in {feature_name} of {category_name}')
            elif(batch_count == 0):
                batch_count = feature_batch_count
                batch_file_array = feature_batch_files
            elif (feature_batch_count != batch_count):
                raise RuntimeError(
                    f'No of batches in {feature_name} of {category_name} does not match with previous batch count')
        batchFiles.append(batch_file_array)


def _read_features(path, selected_files_array):
    featurefiles = os.listdir(path)
    biomer_dfs = []
    feature_df = pd.DataFrame()
    for ff in featurefiles:
        if (ff == 'fragments' or ff == '7mers'):
            continue
        print(ff)
        single_f_df = _read_feature_files(
            path + '/' + ff, featureNames[ff], selected_files_array)
        featureCounts[ff] = len(single_f_df)
        if(feature_df.empty):
            feature_df = single_f_df
        else:
            feature_df = feature_df.merge(single_f_df, how='inner', on='id')
    print('7mers')
    kmer_df = _read_feature_files(
        path + '/7mers', ['7mer-' + str(j) for j in range(8192)], selected_files_array)
    feature_df = feature_df.merge(kmer_df, left_index=True, right_index=True)
    feature_df['id'] = feature_df['id'].astype(np.int32)
    featureCounts['7mers'] = len(kmer_df)
    return feature_df


def _read_feature_files(path, feature_names, selected_files_array):
    fileArrays = []
    for file in tqdm(selected_files_array):
        fileArrays.append(np.genfromtxt(path+"/"+str(file), dtype=np.float64))
    featureArray = np.concatenate(fileArrays)
    feature_df = pd.DataFrame(featureArray, columns=feature_names)
    return feature_df


def _logToFile(msg):
    with open(logfile, 'a') as log:
        log.write(msg+'\n')


def _logToFilewithPrint(msg):
    print(msg)
    with open(logfile, 'a') as log:
        log.write(msg+'\n')
