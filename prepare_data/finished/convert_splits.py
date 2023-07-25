import torch
import pandas as pd

#a = torch.load("/share/klab/datasets/EEG_Visual/block_splits_by_image_all.pth")
a = torch.load("/share/klab/datasets/EEG_Visual/quarter_splits.pth")

train = a['splits'][0]["train"]
print(len(train))
test = a['splits'][0]["test"]
print(len(test))
val = a['splits'][0]["val"]
print(len(val))
print((len(val)) + (len(test)) + (len(train)))

print(train)
print(test)
print(val)

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)
val_df = pd.DataFrame(val)

# create csv to store train/test/val indices for NC

#train_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/split_train.csv", index=False)
#test_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/split_test.csv", index=False)
#val_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/split_val.csv", index=False)

train_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_train.csv", index=False)
test_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_test.csv", index=False)
val_df.to_csv("/share/klab/datasets/EEG_Visual/NeuCube_Format/quarter_split_val.csv", index=False)


