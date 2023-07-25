import torch
import pandas as pd
import glob
import csv
import re
import os
import time

start_time = time.time()

master_path = "/share/klab/datasets/EEG_Visual/"
eeg_range = '55_95_quarter'

if eeg_range == 'raw':
    file = 'eeg_signals_raw_with_mean_std.pth'
    directory = 'Raw'
elif eeg_range == '5_95':
    file = 'eeg_5_95_std.pth'
    directory = '5_95'
elif eeg_range == '14_70':
    file = 'eeg_14_70_std.pth'
    directory = '14_70'
elif eeg_range == '55_95':
    file = 'eeg_55_95_std.pth'
    directory = '55_95'
elif eeg_range == 'raw_quarter':
    file = 'raw_quarter.pth'
    directory = 'Raw_Quarter'
elif eeg_range == '5_95_quarter':
    file = '5_95_quarter.pth'
    directory = '5_95_Quarter'
elif eeg_range == '14_70_quarter':
    file = '14_70_quarter.pth'
    directory = '14_70_Quarter'
elif eeg_range == '55_95_quarter':
    file = '55_95_quarter.pth'
    directory = '55_95_Quarter'

a = torch.load(f"{master_path}{file}")

# Separate out the EEG data
for i in range(len(a['dataset'])):
    image = a['dataset'][i]['image']
    label = a['dataset'][i]['label']
    subject = a['dataset'][i]['subject']
    eeg_data = a['dataset'][i]['eeg']
    df = pd.DataFrame(eeg_data)
    df.to_csv(f"{master_path}NeuCube_Format/{directory}/eeg_{directory}_{i}_image{image}_class{label}_subject{subject}.csv", index=False)

with open(f'{master_path}NeuCube_Format/{directory}/tar_class_labels.csv', 'w') as f:
    pass

ds_size = len(a['dataset'])

for i in range(0, ds_size):
    path = f'{master_path}NeuCube_Format/{directory}/*eeg_{directory}_{i}_*.csv'
    files = glob.glob(path)
    file = files[0]
    with open(file,'r') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]
    with open(f'{master_path}NeuCube_Format/{directory}/sam{i}_eeg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    class_from_filename = re.search(r'(?<=class)(\d+)', file)
    label = class_from_filename.group(1)
    label = [label]
    with open(f'{master_path}NeuCube_Format/{directory}/tar_class_labels.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(label)
    os.remove(file)

print('Conversion Successful')

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time/3600)
elapsed_minutes = int((elapsed_time % 3600)/60)

print (f'Conversion took {elapsed_hours} hours and {elapsed_minutes} minutes.')