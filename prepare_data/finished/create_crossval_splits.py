from neucube.encoder import Delta
import torch
import pandas as pd
from sklearn.model_selection import KFold
import pickle

numfiles = 11965

print(f'Number of files in DS being used: {numfiles}')

filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in range(0,numfiles)]

dfs = []
for filename in filenameslist:
    df = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'+filename, header=None)
    df = df.iloc[:, 20:460]
    df = df.transpose()
    dfs.append(df)

fulldf = pd.concat(dfs)

print(fulldf.shape)

X = torch.tensor(fulldf.values.reshape(numfiles,440,128))
encoder = Delta(threshold=0.8)
X = encoder.encode_dataset(X)

kf = KFold(n_splits=5, shuffle=True, random_state=123)

train_splits = []
test_splits= []

for train_index, test_index in kf.split(X):
    train_splits.append(train_index)
    test_splits.append(test_index)

with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits.pkl', 'wb') as f:
    pickle.dump(train_splits, f)

with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits.pkl', 'wb') as f:
    pickle.dump(test_splits, f)

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

torch.save(splits,'/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth')

splits = torch.load('/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth')

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


