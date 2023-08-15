import torch
import pandas as pd

'''
This file takes a master split file, either the original train/test/val splits from the original comparison models dataset or the train/val/test splits for the 
quartered data, and converts them to csv index files that are later loaded by NeuCube for use in splitting the dataframe used by that network during training.
'''

# load the data file to be converted
#a = torch.load("/share/klab/datasets/EEG_Visual/block_splits_by_image_all.pth") # the original split file
a = torch.load("/share/klab/datasets/EEG_Visual/quarter_splits.pth") # the custom split file for the quartered data

# pull the train split from the .pth file, confirming its size
train = a['splits'][0]["train"]
print(len(train))

# pull the test split from the .pth file, confirming its size
test = a['splits'][0]["test"]
print(len(test))

# pull the validation split from the .pth file, confirming its size
val = a['splits'][0]["val"]
print(len(val))

# confirm the size of the combined splits (should be the full or quartered size of the DS based on which split file was selected)
print((len(val)) + (len(test)) + (len(train)))

# print the splits
print(train)
print(test)
print(val)

# convert the pulled splits to panda dataframes for easier conversion to csv files later
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)
val_df = pd.DataFrame(val)

# create csv to store train/test/val indices for NC
# uncomment the appropriate set of lines below based on whether the full dataset splits or quartered data splits are being used

#train_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/split_train.csv", index=False)
#test_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/split_test.csv", index=False)
#val_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/split_val.csv", index=False)

train_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv", index=False)
test_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv", index=False)
val_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv", index=False)


