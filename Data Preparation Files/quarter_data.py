import torch
import random

'''
This file creates an index file for train/val/test splits which can be used to create a quartered dataset from any of the master dataset files, balanced across subject and 
class.
'''

# specify the location of the master directory where the splits file has been copied
master_path = "/share/klab/datasets/EEG_Visual/"

# specify the name of the splits file (this line should not be changed)
splits_file = 'block_splits_by_image_all.pth'

# specify which EEG filter range (i.e. which dataset) is wanted
eeg_range = '55_95'

# specify the master data file and directory name to be used based on the selected range
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

# load the selected master data file and the splits file
a = torch.load(f"{master_path}{file}")
s = torch.load(f"{master_path}{splits_file}")

# create 6 lists, one for each subject, to hold the index values by subject
sub_lists = [[] for j in range(1,7)]

# append the index values in the DS to the respective subject lists
for i in range(len(a['dataset'])):
    if a['dataset'][i]['subject'] == 1:
        sub_lists[0].append(i)
    elif a['dataset'][i]['subject'] == 2:
        sub_lists[1].append(i)
    elif a['dataset'][i]['subject'] == 3:
        sub_lists[2].append(i)
    elif a['dataset'][i]['subject'] == 4:
        sub_lists[3].append(i)
    elif a['dataset'][i]['subject'] == 5:
        sub_lists[4].append(i)
    elif a['dataset'][i]['subject'] == 6:
        sub_lists[5].append(i)

# create 40 lists, one for each class
lab_lists = [[] for j in range(40)]

# go through the 6 subject lists and sort into the 40 class lists
for i in range (6):
    for value in sub_lists[i]:
        for j in range(40):
            if a['dataset'][value]['label'] == j:
                lab_lists[j].append(value)

# create 40 delete lists to be used to remove 75% of the data from each category
del_lists = [[] for j in range(40)]

# create a master delete list to hold all values to be removed
delete_list = []

# cycle through each of the classes and move 75% of indices for that class to a delete list for that class, which is then appended to the master delete list
for i in range(40):
    del_lists[i] = random.sample(lab_lists[i],int(len(lab_lists[i])*0.75))
    delete_list = delete_list + del_lists[i]

# sort the delete list from high to low (without this line, the program will attempt to delete entries that are no longer present due to renumbering resulting from
# removing values lower in the list; this way, the values will be removed from largest to smallest, with each deletion not affecting the process)
delete_list.sort(reverse=True)

# create a list of all the index values in the entire dataset and confirm its length
pre_list = s['splits'][0]["train"] + s['splits'][0]["val"] + s['splits'][0]["test"]
print(f'original ds is of size {len(pre_list)}')

# create a new list containing all the values in the dataset that are not marked for deletion by the delete list
full_list = [x for x in pre_list if x not in set(delete_list)]

# for every value in the delete list, remove it from its respective split in the splits file and remove it from the dataset
for value in delete_list:

    if value in s['splits'][0]["train"]:
        s['splits'][0]["train"].remove(value)

    elif value in s['splits'][0]["test"]:
        s['splits'][0]["test"].remove(value)

    elif value in s['splits'][0]["val"]:
        s['splits'][0]["val"].remove(value)

    a['dataset'].pop(value)

# create a list containing all the index values in the new splits and print its size
post_list = s['splits'][0]["train"] + s['splits'][0]["val"] + s['splits'][0]["test"]
print(f'post delete ds is of size {len(post_list)}')

# print how much smaller the new quartered splits are than the original, should be roughly 25%
percent = len(post_list)/len(pre_list)
print(f'This is the expected {percent} of the original ds.')

# print out the per-category percentage to confirm that this has been maintained
for i in range(0, 40):
    count = sum([1 for value in full_list if a['dataset'][value]['label'] == i])
    percentage = count / len(full_list) * 100
    print(f"Category {i}: {percentage:.2f}%")

# print the per-subject percentage to confirm that this has been maintained
for i in range(1,7):
    count = sum([1 for value in full_list if a['dataset'][value]['subject'] == i])
    percentage = count / len(full_list) * 100
    print(f"Subject {i}: {percentage:.2f}%")

# save the new quartered dataset file and the new quartered splits file
# note that the .pth files for the quarted data are not actually in training, but are instead used only to create the appropriate csv files for training NeuCube
# when the comparison models train, the quartered splits file already handles reducing the data size to a quarter of the original
torch.save(a, f'{master_path}{eeg_range}_quarterpth')
torch.save(s, f'{master_path}quarter_splits.pth')

