import torch
import pandas as pd
import glob
import csv
import re
import os
import time

eeg_range = '55_95_quarter'

master_path = "/share/klab/datasets/EEG_Visual/"

train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv')
test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv')
val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv')

train_split = train_split.transpose()
train_split = train_split.to_numpy()
train_split = train_split[0]

test_split = test_split.transpose()
test_split = test_split.to_numpy()
test_split = test_split[0]

val_split = val_split.transpose()
val_split = val_split.to_numpy()
val_split = val_split[0]

idx = train_split.tolist() + test_split.tolist() + val_split.tolist()

idx.sort()

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
    file = 'eeg_signals_raw_with_mean_std.pth'
    directory = 'Raw_Quarter'
elif eeg_range == '5_95_quarter':
    file = 'eeg_5_95_std.pth'
    directory = '5_95_Quarter'
elif eeg_range == '14_70_quarter':
    file = 'eeg_14_70_std.pth'
    directory = '14_70_Quarter'
elif eeg_range == '55_95_quarter':
    file = 'eeg_55_95_std.pth'
    directory = '55_95_Quarter'

a = torch.load(f"{master_path}{file}")

with open(f'{master_path}NeuCube_Format/{directory}/tar_class_labels.csv', 'w') as f:
    pass

# Separate out the EEG data
for idx in idx:
    label = a['dataset'][idx]['label']

    with open(f'{master_path}NeuCube_Format/{directory}/tar_class_labels.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(iter([label]))

print('Label Creation Successful')