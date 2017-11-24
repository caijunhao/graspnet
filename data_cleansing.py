import os
path = '/home/caijunhao/data/cmu_patch_datasets/Train/negative'

image_list = os.listdir(os.path.join(path, 'Images'))
with open(os.path.join(path, 'dataInfo.txt'), 'r') as f:
    data_info = f.readlines()
    data_info_list = [line.strip().split(',') for line in data_info]

with open(os.path.join(path, 'datainfo.txt'), 'w') as f:
    for i in range(len(data_info_list)):
        if data_info_list[i][0] in image_list:
            f.write(data_info[i])
