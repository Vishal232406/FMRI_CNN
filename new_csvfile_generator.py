import glob
import nibabel as nib
import pandas as pd
import torch.utils.data
import numpy as np
import csv
import os



mypath = r"E:\IMP\CSCI595\FMRI\CMU"

img_files_path = glob.glob(mypath + "/*.nii")

FILE_ID = []
SITE_ID = []
for i in img_files:
    if i[24:27] == 'CMU':
        j = i[24:37]
        FILE_ID.append(j)
        SITE_ID.append('CMU')
        
    elif i[24:27] == 'NYU':
        j = i[24:35]
        FILE_ID.append(j)
        SITE_ID.append('NYU')
        
    else :
        j = i[24:36]
        FILE_ID.append(j)
        k = i[24:28]
        SITE_ID.append(k.upper())
    
data = pd.read_csv('E:\IMP\CSCI595\FMRI\Phenotypic_V1_0b_preprocessed1.csv')

file_id_column = data.FILE_ID[:]


file_id_array = []
for i in FILE_ID:
    for j in file_id_column:
        if i == j:
            new_data = data[data['FILE_ID'] == i]
            new_dataframe = new_dataframe.append(new_data, ignore_index=True)


new_dataframe.to_csv("NEW_FMRI.csv", index=False)

