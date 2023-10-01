import cv2
import torch
from torch import nn
import numpy as np
from torchvision import transforms
from train import train
import matplotlib.pyplot as plt
from PIL import Image
from dataset.culaneDataset import CULane
from torch.utils.data import DataLoader
class LaneRSC(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        # self.resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        # self.resnet50.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.resnet50.bn1 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.resnet152.fc = torch.nn.Linear(2048, 4*N*2) # 280 for testing cuz 4*N*2, nd N on testing is 35, on training is 25.

        #removing last layer from resnet152 (fc)
        modules = list(self.resnet50.children())[:-1]
        self.resnet50 = nn.Sequential(*modules)

        # freezing the pre-trained backbone, so that we will just train our fc.
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # modules = list(self.mobilenet.children())[:-1]
        # self.mobilenet = nn.Sequential(*modules)
        # print(self.mobilenet.modules)


        # FPN.

        # Add our custom fully connected network
        self.custom_fc = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(1280, 4*N*2)
            nn.Linear(2048, 4 * N * 2)
        )
        # print(self.resnet152.modules)

        self.laneRSC = nn.Sequential(
            self.resnet50,
            # self.mobilenet,
            self.custom_fc
        )


    def forward(self, x):
        return self.laneRSC(x)



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = '/mnt/Travail/DLProjects/RSC/LaneRSC/datasets/culane'
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    N = 25

    model = LaneRSC(N).to(device)

    #load the saved model's state_dict if available
    try:
        model.load_state_dict(torch.load('trained_model.pth'))
        print("Loaded pre-trained model.")
    except FileNotFoundError:
        print("No pre-trained model found. Training from scratch.")

    train_dataset = CULane(data_root, 'train')  # train => N = 25 , test => 35, change it from model.py + train.pyin normalize_img_keypoints!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate)

    train(model, train_loader, BATCH_SIZE, NUM_EPOCHS)

    # Save the trained model after the training phase
    torch.save(model.state_dict(), 'trained_model.pth')

    #GG!

    # # #
    # img_path = "/mnt/Travail/DLProjects/RSC/LaneRSC/datasets/culane/driver_161_90frame/06031610_0866.MP4/04410.jpg"
    # img = Image.open(img_path)
    # # img.show()
    #
    # t = transforms.ToTensor()
    # resize_features = transforms.Resize((300,900))
    # features = t(img)
    # print(features.shape)
    # features = resize_features(features)
    # tensor_to_image = transforms.ToPILImage()
    # tensor_to_image(features).show()
    #
    # features = features.unsqueeze(0)
    # print(features.shape)
    #
    # test  = model(features.to(device))
    # print(test.shape)
    # test = test.squeeze(0)
    #
    # test = test.cpu()
    # test = test.detach().numpy()
    # print(test)
    # img = np.ones((500, 1240, 3), dtype=np.uint8)
    # # img = 255 * img
    # for i in range(0, test.__len__(),2):
    #       img = cv2.circle(img, (test[i], test[i + 1]), radius=0, color=(0, 255, 0), thickness=10)
    #
    # plt.figure(figsize=(16, 18))
    # plt.imshow(img)
    # plt.show()

    #GG!
