import torch
import pandas as pd
import csv

'''
This file creates the NeuCube label file for the quartered dataset based on the quartered indices for the full datasets. Note that this can also be done working with 
the label files created during quartering directly, with some minor modifications to the master NeuCube training file. This method is reproduced here as it was the one
used in the reported training rounds.
'''

# select the desired frequency range (i.e. dataset)
eeg_range = '55_95_quarter'

# define the master directory where the data files are stored
master_path = "/share/klab/datasets/EEG_Visual/"

# load the train/val/test splits for the quartered dataset
train_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv')
test_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv')
val_split = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv')

# convert the splits to numpy arrays; note that this requires transposition as the values are arranged in a vertical pattern in the csv but must be in a
# horizontal pattern for iterating
train_split = train_split.transpose()
train_split = train_split.to_numpy()
train_split = train_split[0]

test_split = test_split.transpose()
test_split = test_split.to_numpy()
test_split = test_split[0]

val_split = val_split.transpose()
val_split = val_split.to_numpy()
val_split = val_split[0]

# combine the lists into a master index list for the quartered DS
idx = train_split.tolist() + test_split.tolist() + val_split.tolist()

# sort this list from low to high
idx.sort()

# set the split file and directory based on the frequency range selected
# note that here, the code uses the master dataset .pth files instead of the quartered files, though with minor code modifications, this can be done as well
# (note that in this case, the code in the master NeuCube training file will need to be updated accordingly)
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

# load the respective data file
a = torch.load(f"{master_path}{file}")

# create the label file for quartering in the appropriate directory
with open(f'{master_path}NeuCube_Format/{directory}/tar_class_labels.csv', 'w') as f:
    pass

# pull the label for each segment in the quartered list from the dataset and add it to the label file
for idx in idx:
    label = a['dataset'][idx]['label']

    with open(f'{master_path}NeuCube_Format/{directory}/tar_class_labels.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(iter([label]))

print('Label Creation Successful')