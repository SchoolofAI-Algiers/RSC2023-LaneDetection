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
from fpn import FPN
class LaneRSC(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        # self.resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)


        #removing last layer from resnet152 (fc)
        modules = list(self.resnet50.children())[:-1]
        self.resnet50 = nn.Sequential(*modules,
                                      nn.Conv2d(2048, 256, kernel_size=1)
                                      )

        # freezing the pre-trained backbone, so that we will just train our fc.
        for param in self.resnet50.parameters():
            param.requires_grad = False



        # FPN.

        self.fpn = FPN(
            in_channels= [256] ,
            out_channels= 256,
            num_outs= 1
        )
        # Add our custom fully connected network
        self.custom_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4 * N * 2)
        )



    def forward(self, x):
        x = self.resnet50(x)
        x = x.unsqueeze(1).expand(-1, 256, -1, -1, -1)
        print(x.shape)
        x = self.fpn(x)
        x = x[0]
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0).expand(55, -1, -1, -1)
        print(x.shape)
        x = self.custom_fc(x)
        return x



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = '/mnt/Travail/DLProjects/RSC/LaneRSC/datasets/culane'
    # BATCH_SIZE = 8
    BATCH_SIZE = 55
    NUM_EPOCHS = 3
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
    torch.save(model.state_dict(), 'trained_models/trained_model.pth')

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
