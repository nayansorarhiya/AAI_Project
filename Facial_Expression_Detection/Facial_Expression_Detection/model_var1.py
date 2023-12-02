import torch.nn as nn
from Classification import ImageClassificationBase


class myCNNModel_var1(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conversion_block(in_channels, 64)
        self.conv2 = conversion_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conversion_block(128, 128), conversion_block(128, 128))

        self.conv3 = conversion_block(128, 256, pool=True)
        self.conv4 = conversion_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conversion_block(512, 512), conversion_block(512, 512))
        self.conv5 = conversion_block(512, 512, pool=True)

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
def conversion_block(in_channels, out_channels, pool=False):
    kernel_size = 3
    
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)