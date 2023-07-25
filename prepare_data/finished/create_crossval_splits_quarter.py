from neucube.encoder import Delta
import torch
import pandas as pd
from sklearn.model_selection import KFold
import pickle
import numpy as np

a = torch.load('/share/klab/datasets/EEG_Visual/quarter_splits.pth')

idx = []

for value in a['splits'][0]['train']:
    idx.append(value)

for value in a['splits'][0]['test']:
    idx.append(value)

for value in a['splits'][0]['val']:
    idx.append(value)

idx.sort()

filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in idx]

dfs = []
for filename in filenameslist:
    df = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'+filename, header=None)
    df = df.iloc[:, 20:460]
    df = df.transpose()
    dfs.append(df)

fulldf = pd.concat(dfs)

print(fulldf.shape)

X = torch.tensor(fulldf.values.reshape(2993,440,128))
encoder = Delta(threshold=0.8)
X = encoder.encode_dataset(X)

kf = KFold(n_splits=5, shuffle=True, random_state=123)

train_splits = []
test_splits= []
train_splits_n = []
test_splits_n = []

for train_index, test_index in kf.split(X):
    train_splits_n.append(train_index)
    test_splits_n.append(test_index)
    train_index = [idx[i] for i in np.array(train_index)]
    test_index = [idx[i] for i in np.array(test_index)]
    train_splits.append(train_index)
    test_splits.append(test_index)

with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits_quarter.pkl', 'wb') as f:
    pickle.dump(train_splits_n, f)

with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits_quarter.pkl', 'wb') as f:
    pickle.dump(test_splits_n, f)

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

torch.save(splits,'/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth')

splits = torch.load('/share/klab/datasets/EEG_Visual/NeuCube_Splits_Quarter.pth')

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

print('with sizes')
print(len(train))
print(len(test))


