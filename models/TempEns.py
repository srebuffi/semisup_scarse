import torch.nn as nn
import torch.nn.functional as F


class TempEns(nn.Module):
    def __init__(self, input_dim=3, output_dim=10):
        super(TempEns, self).__init__()

        batchNorm_momentum = 0.999
        # batchNorm_momentum = 0.0
        leak = 0.1
        drop_p = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(drop_p)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(drop_p)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.LeakyReLU(negative_slope=leak),
        )

        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 6)
        x = self.fc(x.view(x.size(0), -1))
        return x
