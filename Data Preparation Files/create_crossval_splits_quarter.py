from neucube.encoder import Delta
import torch
import pandas as pd
from sklearn.model_selection import KFold
import pickle
import numpy as np

# load the splits file for the quartered datasets
a = torch.load('/share/klab/datasets/EEG_Visual/quarter_splits.pth')

# load the splits into a master list and sort from smallest to lowest
idx = []

for value in a['splits'][0]['train']:
    idx.append(value)

for value in a['splits'][0]['test']:
    idx.append(value)

for value in a['splits'][0]['val']:
    idx.append(value)

idx.sort()

# lines 29 - 69 largely taken from the original NeuCube code, updated to create appropriate split files for the comparison networks and NeuCube, as the index values differ
# between the two in this case as NeuCube starts work in the 5K training paradigm with an already quartered dataset (and thus a smaller index range)

# create a list of the filenames contained in the quartered dataset
filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in idx]

# create a dataframe from the quartered files, taking the data corresponding to 20-460ms post stimulus and transposing it into the expected format for NeuCube (time x features
# versus features x time)
dfs = []
for filename in filenameslist:
    df = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'+filename, header=None)
    df = df.iloc[:, 20:460]
    df = df.transpose()
    dfs.append(df)
fulldf = pd.concat(dfs)

print(fulldf.shape)

# create a tensor of the size 2993 (one quarter of the ds) x 440 (the length of the EEG segments) x 128 (the number of features - here channels)
X = torch.tensor(fulldf.values.reshape(2993,440,128))
encoder = Delta(threshold=0.8)
X = encoder.encode_dataset(X)

# load the fold creater with the KFold command
kf = KFold(n_splits=5, shuffle=True, random_state=123)

# create lists to hold the indices for the quartered DS
# note that here, we create two separate index list pairs because the NeuCube training code will load a smaller dataset with a correspondingly smaller index range
# when running in the 5K paradigm for the quartered data, whereas the comparison networks will pull the respective files from the full datasets (with the full index range)
train_splits = []
test_splits= []
train_splits_n = []
test_splits_n = []

for train_index, test_index in kf.split(X):

    # save the splits with the shorter index range for NeuCube
    train_splits_n.append(train_index)
    test_splits_n.append(test_index)

    # save the splits with the full index range for the comparison models
    train_index = [idx[i] for i in np.array(train_index)]
    test_index = [idx[i] for i in np.array(test_index)]
    train_splits.append(train_index)
    test_splits.append(test_index)

# save the splits to be used by NeuCube as pkl files for later loading
with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits_quarter.pkl', 'wb') as f:
    pickle.dump(train_splits_n, f)

with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits_quarter.pkl', 'wb') as f:
    pickle.dump(test_splits_n, f)

# create a dictionary containing the splits for all folds in a format that can be used by the comparison networks
splits = {
    'splits': {
        0: {
            'train': train_splits[0],
            'test': test_splits[0]
        },

        1: {
            'train': train_splits[1],
            'test': test_splits[1]
        },

        2: {
            'train': train_splits[2],
            'test': test_splits[2]
        },

        3: {
            'train': train_splits[3],
            'test': test_splits[3]
        },

        4: {
            'train': train_splits[4],
            'test': test_splits[4]
        }
    }
}

# save the splits to the .pth file the comparison networks will use to train
torch.save(splits,'/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth')

# reload the quartered splits file to check it
splits = torch.load('/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth')

# print out info on the quartered splits file
print('The sections of the dataset file are: ')
print(splits.keys())

length = len(splits['splits'])
print('There are this many splits in the file:')
print(length)

print('the first set of splits is')
train = splits['splits'][0]["train"]
test = splits['splits'][0]["test"]

print(train)
print(test)

# confirm the splits are of the expected size (for a quartered DS)
print('with sizes')
print(len(train))
print(len(test))


