import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset.culaneDataset import CULane
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# show_targets - a simple function that draw a list of keypoints on an image.
# @exist_list: list of keypoints (list of lists, each sublist represent a lane).
# @img: the image i want to display the keypoints on.
def show_targets(exist_list, img):
    for j in range(0, exist_list.__len__()):
        for i in range(0, exist_list[j].__len__(), 2):
            img = cv2.circle(img, (exist_list[j][i], exist_list[j][i + 1]), radius=0, color=(0, 255, 0), thickness=10)
    plt.figure(figsize=(16, 18))
    plt.imshow(img)
    plt.show()


# normalize_img_keypoints - a simple function that
# unifies a list of lanes for an image to one structure (4, 50), 4 lanes nd 50 ints that
# represent 50 keypoitns au max.
# @img_keypoints: a list of lists, each sublist represent a lane, initially each image
# has its own structure for this set.
def normalize_img_keypoints(img_keypoints):
    N = 25  # nombre de points maximal that represent a lane.

    if (len(img_keypoints) < 4):  # fill an image with 4 lanes.
        img_keypoints += [[]] * (4 - len(img_keypoints))

    for lane in img_keypoints:  # fill a lane with 25 points.
        if len(lane) / 2 != N:
            lane += [-1] * (2 * N - len(lane))  # fill the rest of lane with zeros.
            # print(len(lane))
            # or np.nan or -1. -100


def normalize_convert_images_to_tensor(batch_images):


    batch_images = np.array(batch_images)
    batch_images = batch_images / 255  # normalized

    #TODO:  modify image size.

    return torch.tensor(batch_images)


def generate_batch_targets(batch_img_paths):
    # iterate over a batch, and aggregate all 8 image keypoints in one set.
    batch_targets = []
    for image_id in range(BATCH_SIZE):
        img_path = batch_img_paths[image_id]
        anno_path = img_path[:-3] + 'lines.txt'
        img_keypoints = []  # image ground truth
        with open(anno_path) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                img_keypoints.append([int(eval(x)) for x in l[2:]])

        normalize_img_keypoints(img_keypoints)
        # print(len(img_keypoints)) # 4.
        batch_targets.append(img_keypoints)

    # checks - debugging
    print(len(batch_targets))
    for i in range(BATCH_SIZE):
        print(len(batch_targets[i])) #....4
        for j in range(4):
            print(len(batch_targets[i][j]))#....50
    print(batch_targets)

    batch_targets = np.array(batch_targets, dtype=int)
    return torch.tensor(batch_targets)





#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#HyperParams

learning_rate = 0.001
NUM_EPOCHS = 1
BATCH_SIZE = 8

data_root = '/mnt/Travail/DLProjects/RSC/LaneRSC/datasets/culane'

train_dataset = CULane(data_root, 'train') # train => N = 25 , test => 35
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate)
#init model

# resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#  model = CNN().to(device)


#Loss nd optimizer






def train(model):
    #FIXME
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train a model
    for epoch in range(NUM_EPOCHS):
        for batch_id, batch in enumerate(train_loader):
            #each batch is a CULANE_sample (a dict with a list of images, images_paths, that's all what we need from him. )
            # in one batch there are :
            # batch['img']: a list of 32 images(590x1640)
            # batch['img_name']:a list of SIZE image paths.
            batch_images = normalize_convert_images_to_tensor(batch['img'])
            # batch_images = batch_images.reshape(batch_images.shape[0], -1)
            # print(batch_images)
            batch_images = torch.movedim(batch_images, 1, -1)
            batch_images = torch.movedim(batch_images, 1, -1)
            # batch_images = batch_images.resize()
            #input of our model : ( 3, 590, 1640 )
            print(batch_images.shape)  # (32, 3, 590, 1640)
            print(batch['img_name'])
            batch_targets = generate_batch_targets(batch['img_name'])
            print(batch_targets.shape) # ouput of our model (32, 4, 50)
            # break
            batch_images = batch_images.to(device=device)
            batch_targets = batch_targets.to(device=device)

            # print(batch_images)
            scores = model(batch_images.float())
            print(scores.shape) #[32, 4, 50 ]?

            loss = criterion(scores, batch_targets.float())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # Sg or adam step ( updating weights )
            optimizer.step()

            # # testing the targets after reshaping em
            # img = np.ones((590, 1640, 3), dtype=np.uint8)
            # img = 255 * img
            # show_targets(batch_targets[0], img)
