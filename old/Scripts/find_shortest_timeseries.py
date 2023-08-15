import os
import pandas as pd

directory = '/share/klab/datasets/EEG_Visual/EEG_CSV/'
min_cols = 500

for filename in os.listdir(directory):
    if (filename.endswith('.csv')) and ('tar' not in filename):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        num_cols = df.shape[1]
        if num_cols < min_cols:
            min_cols = num_cols

print(f'The file with the shortest time series is {min_cols} columns long.')