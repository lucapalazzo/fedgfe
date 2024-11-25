import pandas as pd
import os
import re
from pathlib import Path

dataset_path = 'dataset'
datafile = dataset_path + '/Data Chest X-Ray RSUA (Validated)/Split_Data_RSUA_Paths_k3.xlsx'

data = pd.read_excel(datafile, index_col=0)

labels = []
for index,col in data.iterrows():
    splitted = col['images_path'].split('/')
    label = splitted[-3]
    if label not in labels:
        labels.append(label)
        print("Added %d col %s" %(index, label))
    mask = col['masks_path']
    path = Path(dataset_path+'/'+col['images_path'])
    # print("index %d col %s %s" %(index, mask, path))
    if not path.exists():
        print("Path %s does not exist" %path)
    else:
        image_length = os.path.getsize(path)
        print ( "File %s size %d" %(path, image_length))

for label_id in range(len(labels)):
    print("Id %d %s" %(label_id, labels[label_id]))