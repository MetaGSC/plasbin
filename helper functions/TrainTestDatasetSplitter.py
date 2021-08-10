import os
import pandas as pd
from pathlib import Path
import shutil


def splitTrainTestDatasets(CurrDir,NewDir, missReportDir, csvPath, testFraction):
    df_split = pd.read_csv(csvPath)
    df_split.apply(lambda x: _moveDataFiles(
        x, CurrDir, NewDir, missReportDir), axis=1)
    df_split.apply(lambda x: _moveLabelFiles(
        x, CurrDir, NewDir, missReportDir), axis=1)


def _moveDataFiles(row, CurrDir,NewDir, missReportDir):
    filename = str(row['index'])+'.txt.gz'
    data_array = os.listdir(CurrDir+'/Data')
    if(int(row.name) % 10 == 0):
        print(row.name)
    for data_folder in data_array:
        try:
            # os.rename(CurrDir+"/Data/"+data_folder+"/"+filename, NewDir +
            #         "/"+str(row['type'])+"/Data/"+data_folder+"/"+filename)
            shutil.copyfile(CurrDir+"/Data/"+data_folder+"/"+filename, NewDir +
                            "/"+str(row['type'])+"/Data/"+data_folder+"/"+filename)
        except:
            with open(missReportDir+'/misses.txt', 'a') as miss_file:
                miss_file.write(data_folder+" - "+filename+"\n")
            return


def _moveLabelFiles(row, CurrDir, NewDir, missReportDir):
    filename = str(row['index'])+'.txt.gz'
    data_array = os.listdir(CurrDir+'/Data')
    if(int(row.name) % 10 == 0):
        print(row.name)
    try:
        # os.rename(CurrDir+"/Label/"+filename, NewDir +
        #         "/"+str(row['type'])+"/Label/"+filename)
        shutil.copyfile(CurrDir+"/Label/"+filename, NewDir +
                        "/"+str(row['type'])+"/Label/"+filename)
    except:
        with open(missReportDir+'/misses.txt', 'a') as miss_file:
            miss_file.write("Label - "+filename+"\n")
        return


def divideDatasets(csvPath, testFraction, finalCSVsDir, batchSize):
    df_class = pd.read_csv(csvPath)
    df_cat = df_class.groupby(df_class['class']).sample(frac=testFraction)
    df_cat['type'] = 'Test'
    df_class = df_class.merge(df_cat, how="outer", on=[
                              'index', 'class']).fillna("Train")
    print(df_class)
    batchCount = (len(df_class)/batchSize)+1
    for b in range(int(batchCount)):
      if(b < batchCount-1):
        df_batch = df_class[(batchSize*b):(batchSize*(b+1))]
      else:
        df_batch = df_class[(batchSize*b)::]
      df_batch.to_csv(finalCSVsDir+'/batch_'+str(b)+'.csv', index=False)


def createFolders(mainDir, dataFolderArray):
    os.mkdir(mainDir+'/Test')
    os.mkdir(mainDir+'/Test/Data')
    os.mkdir(mainDir + '/Test/Label')
    os.mkdir(mainDir+'/Train')
    os.mkdir(mainDir+'/Train/Data')
    os.mkdir(mainDir+'/Train/Label')
    for folder in dataFolderArray:
        os.mkdir(mainDir+'/Test/Data/'+folder)
        os.mkdir(mainDir+'/Train/Data/'+folder)
