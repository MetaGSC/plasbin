import os
import gzip
import shutil
import re
from tqdm import tqdm

def moveFiles(CurrDir,NewDir):
    for f in tqdm(os.scandir(CurrDir)):
        try:
            os.rename(CurrDir+"/"+f.name, NewDir +
                    "/"+f.name)
        except:
            return


def copyFiles(CurrDir, NewDir):
    for f in tqdm(os.scandir(CurrDir)):
        try:
            shutil.copyfile(CurrDir+"/"+f.name, NewDir + "/" + f.name)
        else:
            break

def gzipFilesInFolder(Dir,NewDir):
    count = 0
    dcount = 0
    for f in tqdm(os.scandir(Dir)):
        if re.search("\.txt$", f.name):
            with open(Dir+'/'+f.name, 'rb') as f_in, gzip.open(NewDir+'/'+f.name+'.gz', 'wb') as f_out:
                f_out.writelines(f_in)
                count += 1
        else:
            dcount += 1
    print(f'{count} files zipped, {dcount} ignored')
