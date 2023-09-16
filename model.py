import cv2
import torch.nn
import numpy as np
from torchvision import transforms
from train import train
import matplotlib.pyplot as plt
from PIL import Image
class LaneRSC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # self.resnet50.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.resnet50.bn1 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.resnet50.fc = torch.nn.Linear(2048, 4*50)

        # self.conv1 = torch.nn.Conv2d()
        print(self.resnet50.modules)

    def forward(self, x):

        return self.resnet50(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneRSC().to(device)
    train(model)


    #
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

    #GG with a single image! kanet expected 2D or 3D input apres dert image whda brk


    # probably lzm tne9es input size dok, computation tel3o bzf et cuda t3emer....