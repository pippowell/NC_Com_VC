import pandas as pd

numfiles = 20

train_split = pd.read_csv('/Users/powel/PycharmProjects/NeuCube_EEGCN_Visual/split_train.csv')
train_split = train_split.iloc[:numfiles, :]
train_split = train_split.transpose()
train_split = train_split.to_numpy()
train_split = train_split[0]
train_split = train_split[train_split<numfiles]
print(train_split)

test_split = pd.read_csv('/Users/powel/PycharmProjects/NeuCube_EEGCN_Visual/split_test.csv')
test_split = test_split.iloc[:numfiles, :]
test_split = test_split.transpose()
test_split = test_split.to_numpy()
test_split = test_split[0]
test_split = test_split[test_split<numfiles]
print(test_split)

val_split = pd.read_csv('/Users/powel/PycharmProjects/NeuCube_EEGCN_Visual/split_val.csv')
val_split = val_split.iloc[:numfiles, :]
val_split = val_split.transpose()
val_split = val_split.to_numpy()
val_split = val_split[0]
val_split = val_split[val_split<numfiles]
print(val_split)
