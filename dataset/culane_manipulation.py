
import cv2
import matplotlib.pyplot as plt
import torch

from culaneDataset import CULane
from torch.utils.data import DataLoader


#constants
data_root = '/mnt/Travail/DLProjects/RSC/LaneRSC/datasets/culane'


train_dataset = CULane(data_root, 'train')
test_dataset = CULane(data_root, 'test')

# print(train_dataset.__len__())
# print(test_dataset.__len__())


#Init Dataset loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=test_dataset.collate, num_workers=8)


def show_targets(exist_list, img):
    for j in range(0, exist_list.__len__()):
        for i in range(0, exist_list[j].__len__(), 2):
            img = cv2.circle(img, (exist_list[j][i], exist_list[j][i + 1]), radius=0, color=(0, 255, 0), thickness=10)

    plt.figure(figsize=(16, 18))
    plt.imshow(img)
    plt.show()


#show a sample x from culane, (x ~ [0-3376] for testing dataset / x ~  [0-18232] for training dataset )
def show_culane_sample(Dataset, idx):
    img_path = Dataset.__getitem__(idx)['img_name']
    print(img_path)
    img = plt.imread(img_path)
    anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
    exist_list = [] # contains [x, y] for all key points that belong to lanes in the image...all annotations.
    with open(anno_path) as f:
        for line in f:
            line = line.strip()
            l = line.split(" ")
            exist_list.append([int(eval(x)) for x in l[2:]])
    print(exist_list)
    show_targets(exist_list, img)





# show_culane_sample(train_dataset, 1800)
# show_culane_sample(test_dataset, 1800)




