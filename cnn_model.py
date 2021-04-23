import torch.nn.functional as F
import torch
from torch import nn


device = ("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()

        self.convolutional_layer = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=48, kernel_size=(8, 8, 8), dilation=(3, 3, 3), stride=(2, 2, 2),
                      padding=(0, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(in_channels=48, out_channels=86, kernel_size=(8, 8, 8), dilation=(2, 2, 2), stride=(2, 2, 2),
                      padding=(0, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(in_channels=86, out_channels=120, kernel_size=(3, 6, 3), dilation=(1, 1, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.ReLU()
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=120, out_features=80),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=24),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=1)
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)

        x = F.softmax(x, dim=1)
        return x


