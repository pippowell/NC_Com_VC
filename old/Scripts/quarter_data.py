import torch
import random

master_path = "/share/klab/datasets/EEG_Visual/"

splits_file = 'block_splits_by_image_all.pth'

eeg_range = '55_95'

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

a = torch.load(f"{master_path}{file}")
s = torch.load(f"{master_path}{splits_file}")

sub_lists = [[] for j in range(1,7)]

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

lab_lists = [[] for j in range(40)]

for i in range (6):
    for value in sub_lists[i]:
        for j in range(40):
            if a['dataset'][value]['label'] == j:
                lab_lists[j].append(value)

del_lists = [[] for j in range(40)]
delete_list = []

for i in range(40):
    del_lists[i] = random.sample(lab_lists[i],int(len(lab_lists[i])*0.75))
    delete_list = delete_list + del_lists[i]

delete_list.sort(reverse=True)

pre_list = s['splits'][0]["train"] + s['splits'][0]["val"] + s['splits'][0]["test"]
print(f'original ds is of size {len(pre_list)}')

full_list = [x for x in pre_list if x not in set(delete_list)]

for value in delete_list:

    if value in s['splits'][0]["train"]:
        s['splits'][0]["train"].remove(value)

    elif value in s['splits'][0]["test"]:
        s['splits'][0]["test"].remove(value)

    elif value in s['splits'][0]["val"]:
        s['splits'][0]["val"].remove(value)


post_list = s['splits'][0]["train"] + s['splits'][0]["val"] + s['splits'][0]["test"]
print(f'post delete ds is of size {len(post_list)}')

percent = len(post_list)/len(pre_list)
print(f'This is the expected {percent} of the original ds.')

for i in range(0, 40):
    count = sum([1 for value in full_list if a['dataset'][value]['label'] == i])
    percentage = count / len(full_list) * 100
    print(f"Category {i}: {percentage:.2f}%")

for i in range(1,7):
    count = sum([1 for value in full_list if a['dataset'][value]['subject'] == i])
    percentage = count / len(full_list) * 100
    print(f"Subject {i}: {percentage:.2f}%")

torch.save(s, f'{master_path}quarter_splits.pth')

