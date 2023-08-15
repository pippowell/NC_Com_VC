from neucube.encoder import Delta
import torch
import pandas as pd
from sklearn.model_selection import KFold
import pickle

# specify the number of EEG segment files being used, here the full DS (11965)
# lines 10 - 43 largely taken from the original NeuCube code

numfiles = 11965

print(f'Number of files in DS being used: {numfiles}')

# assemble the filenames in a master filename list
filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in range(0,numfiles)]

# create a dataframe containing all the segments in the list, selecting the columns (data) for 20 to 460 ms post stimulus
# note that the data is transposed here as NeuCube expects the data in the format time x features instead of features x time
dfs = []
for filename in filenameslist:
    df = pd.read_csv('/share/klab/datasets/EEG_Visual/NeuCube_Format/Raw/'+filename, header=None)
    df = df.iloc[:, 20:460]
    df = df.transpose()
    dfs.append(df)
fulldf = pd.concat(dfs)
print(fulldf.shape)

# create a torch sensor reshaping the dataframe into the format 11965, 440, 128 and encode the dataset
X = torch.tensor(fulldf.values.reshape(numfiles,440,128))
encoder = Delta(threshold=0.8)
X = encoder.encode_dataset(X)

# load the fold creater using the KFold function
kf = KFold(n_splits=5, shuffle=True, random_state=123)

# create dictionaries to hold the train/test splits for each fold
train_splits = []
test_splits= []

# call the fold creater on the DS and record the train/test splits
for train_index, test_index in kf.split(X):
    train_splits.append(train_index)
    test_splits.append(test_index)

# save the train/test splits in a pkl file for later loading by NeuCube when training in the 5K cross validation paradigm
with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/train_k_splits.pkl', 'wb') as f:
    pickle.dump(train_splits, f)

with open('/share/klab/datasets/EEG_Visual/NeuCube_Format/test_k_splits.pkl', 'wb') as f:
    pickle.dump(test_splits, f)

# create a dictionary of the shape needed by the comparison models as their splits file, with each entry being the train/test splits for one fold
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

# save the dictionary as the splits .pth file needed by the comparison models when training in the 5K paradigm with the full data
torch.save(splits,'/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth')

# load the splits file to check the success of the splitting
splits = torch.load('/share/klab/datasets/EEG_Visual/NeuCube_Splits.pth')

# print information on the splits file and its indexing of the DS
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

# confirm the sizes of the splits make sense for the full DS
print('with sizes')
print(len(train))
print(len(test))


