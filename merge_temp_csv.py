import numpy as np
import pandas as pd
import os
from pathlib import Path

folder_path = Path("/home/tvhahn/Tool-Wear-Files/temp_csv")

i = 0
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        if i == 0:
            file_path = folder_path / file           
            df = pd.read_csv(file_path, na_filter=False) # load csv and replace NaNs with blanks
            i += 1
        else:
            file_path = folder_path / file
            df = df.append(pd.read_csv(file_path, na_filter=False))
        print(file)

# replace blanks in "failed" column with zeros
df['failed'] = df['failed'].replace(to_replace='',value=0).astype(dtype='int16')

# save as a new csv
df.to_csv(('low_level_labels_created_2020.01.30.csv'),index=False)