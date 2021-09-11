import os
import pandas as pd
from pathlib import Path
import shutil


def move_and_split_datasets(CurrentDirectory, NewDirectory, InfoCSVpath, copyFiles=False, zipped=True, testFraction = 0.2, batchSize=50000):
    feature_array = os.listdir(CurrentDirectory+'/Data')
    _create_folders(NewDirectory, feature_array)
    _divide_datasets(InfoCSVpath, testFraction, CurrentDirectory+'/BatchCSVs', batchSize)
    print(f'Data info divided into batches of {batchSize}')
    batchList = os.listdir(CurrentDirectory+'/BatchCSVs')
    for k in range(len(batchList)):
        csv_name = CurrentDirectory+'/BatchCSVs/'+ batchList[k]
        split_train_test_datasets(
            CurrentDirectory, NewDirectory, csv_name, copyFiles, zipped)
        print(f'Batch {k+1} out of {len(batchList)} processed')
    

def split_train_test_datasets(CurrDir, NewDir, csvPath, missReportDir=None,  copyFiles=False, zipped=True):
    if missReportDir is None:
        missReportDir = NewDir
    df_split = pd.read_csv(csvPath)
    df_split.apply(lambda x: _move_data_files(
        x, CurrDir, NewDir, missReportDir, copyFiles, zipped, len(df_split)), axis=1)
    df_split.apply(lambda x: _move_label_files(
        x, CurrDir, NewDir, missReportDir, copyFiles, zipped, len(df_split)), axis=1)


def _move_data_files(row, CurrDir, NewDir, missReportDir, copyFiles, zipped, length):
    
    filename = str(row['index'])+ ('.txt.gz' if zipped else '.txt')
    data_array = os.listdir(CurrDir+'/Data')
    if((int(row.name) % 10 == 0 and int(row.name) < 100) or int(row.name) % 5000 == 0):
        print(f'{row.name} data files out of {length} {'copied' if copyFiles else 'moved'}')
    for data_folder in data_array:
        try:
            if( not copyFiles):
                os.rename(CurrDir+"/Data/"+data_folder+"/"+filename, NewDir +
                        "/"+str(row['type'])+"/Data/"+data_folder+"/"+filename)
            else:
                shutil.copyfile(CurrDir+"/Data/"+data_folder+"/"+filename, NewDir +
                                "/"+str(row['type'])+"/Data/"+data_folder+"/"+filename)
        except:
            with open(missReportDir+'/misses.txt', 'a') as miss_file:
                miss_file.write(data_folder+" - "+filename+"\n")
            return


def _move_label_files(row, CurrDir, NewDir, missReportDir, copyFiles, zipped, length):

    filename = str(row['index']) + ('.txt.gz' if zipped else '.txt')
    if((int(row.name) % 10 == 0 and int(row.name) < 100) or int(row.name) % 1000 == 0):
        print(f'{row.name} label files out of {length} {'copied' if copyFiles else 'moved'}')
    try:
        if( not copyFiles):
            os.rename(CurrDir+"/Label/"+filename, NewDir +
                                "/"+str(row['type'])+"/Label/"+filename)
        else:
            shutil.copyfile(CurrDir+"/Label/"+filename, NewDir +
                            "/"+str(row['type'])+"/Label/"+filename)
    except:
        with open(missReportDir+'/misses.txt', 'a') as miss_file:
            miss_file.write("Label - "+filename+"\n")
        return


def _divide_datasets(csvPath, testFraction, finalCSVsDir, batchSize):
    df_class = pd.read_csv(csvPath)
    df_cat = df_class.groupby(df_class['label_number']).sample(frac=testFraction)
    df_cat['type'] = 'Test'
    df_class = df_class.merge(df_cat, how="outer", on=[
                              'Unnamed: 0','label', 'label_number','file','id']).fillna("Train")
    df_class.to_csv(finalCSVsDir+'/completeSplit.csv', index=False)
    batchCount = (len(df_class)/batchSize)+1
    for b in range(int(batchCount)):
      if(b < batchCount-1):
        df_batch = df_class[(batchSize*b):(batchSize*(b+1))]
      else:
        df_batch = df_class[(batchSize*b)::]
      df_batch.to_csv(finalCSVsDir+'/batch_'+str(b)+'.csv', index=False)


def _create_folders(mainDir, dataFolderArray):
    os.mkdir(mainDir+'/BatchCSVs')
    os.mkdir(mainDir+'/Test')
    os.mkdir(mainDir+'/Test/Data')
    os.mkdir(mainDir + '/Test/Label')
    os.mkdir(mainDir+'/Train')
    os.mkdir(mainDir+'/Train/Data')
    os.mkdir(mainDir+'/Train/Label')
    for folder in dataFolderArray:
        os.mkdir(mainDir+'/Test/Data/'+folder)
        os.mkdir(mainDir + '/Train/Data/' + folder)
