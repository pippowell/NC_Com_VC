import torch
import pandas as pd

a = torch.load("/Users/powel/PycharmProjects/NeuCube_EEGCN_Visual/data/eeg_5_95_std.pth")

print('The sections of the dataset file are: ')
print(a.keys())

print('An example from the dataset is')
first_entry = a['dataset'][0]
print(first_entry)

print('There are this many entries in the dataset dictionary ')
length = len(a['dataset'])
print(length)

print('The EEG data is of size')
first_entry = a['dataset'][0]['eeg']
print(first_entry.shape)

print('An example from the labels is')
first_entry = a['labels'][0]
print(first_entry)

print('there are this many subjects')
num_subjects = len(set([a['dataset'][i]['subject'] for i in range(len(a['dataset']))]))
print(num_subjects)

print('There are this many entries in the labels dictionary ')
length = len(a['labels'])
print(length)

print('An example from the images is')
first_entry = a['images'][0]
print(first_entry)

print('There are this many entries in the images dictionary ')
length = len(a['images'])
print(length)

num_subjects = len(set([a['dataset'][i]['subject'] for i in range(len(a['dataset']))]))

for i in range(0, 40):
    count = sum([1 for j in range(len(a['dataset'])) if a['dataset'][j]['label'] == i])
    percentage = count / len(a['dataset']) * 100
    print(f"Category {i}: {percentage:.2f}%")

for i in range(1,7):
    count = sum([1 for j in range(len(a['dataset'])) if a['dataset'][j]['subject'] == i])
    percentage = count / len(a['dataset']) * 100
    print(f"Subject {i}: {percentage:.2f}%")