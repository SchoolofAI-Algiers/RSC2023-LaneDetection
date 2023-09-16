import torch.nn

from train import train


class LaneRSC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.resnet50.bn1 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.resnet50.fc = torch.nn.Linear(2048, 4*50)
        print(self.resnet50.modules)

    def forward(self, x):
        return self.resnet50(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneRSC().to(device)
    train(model)