import torch
import torch.nn as nn
import torch.nn.functional as F

#### my
class Discriminator_HS(nn.Module):
    def __init__(self, num_conv_block=4):
        super(Discriminator_HS, self).__init__()
        block = []

        in_channels = 31
        out_channels = 128

        for _ in range(num_conv_block):
            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3),
                      #nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU()
                      ]
            in_channels = out_channels

            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3, 2),
                      nn.LeakyReLU()]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, out_channels, 3,1),
                  nn.LeakyReLU(0.2),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(out_channels, out_channels, 3,1),
                  ]

        self.feature_extraction = nn.Sequential(*block)


        self.classification = nn.Sequential(
            nn.Linear(4096, 256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x

class Discriminator_PAN(nn.Module):
    def __init__(self, num_conv_block=4):
        super(Discriminator_PAN, self).__init__()
        block = []

        in_channels = 1
        out_channels = 64

        for _ in range(num_conv_block):
            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3),
                      #nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU()
                      ]
            in_channels = out_channels

            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3, 2),
                      nn.LeakyReLU()]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.Conv2d(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(out_channels, out_channels, 3),
                  ]

        self.feature_extraction = nn.Sequential(*block)

        #self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

        self.classification = nn.Sequential(
            nn.Linear(8192, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x




#### ESRGAN
# class Discriminator_HS(nn.Module):
#     def __init__(self):
#         super(Discriminator_HS, self).__init__()
#         self.in_channels = 31
#
#         def discriminator_block(in_filters, out_filters, first_block=False):
#             layers = []
#             layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
#             if not first_block:
#                 layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
#             layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         layers = []
#         in_filters = self.in_channels
#         for i, out_filters in enumerate([128, 256, 512, 1024]):
#             layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
#             in_filters = out_filters
#
#         layers += [nn.Conv2d(in_filters, in_filters, 3, 1, 1),
#                    nn.LeakyReLU(0.2),
#                    nn.Conv2d(in_filters, out_filters, 3, 1, 1)]
#         self.feature_extraction = nn.Sequential(*layers)
#         self.classification = nn.Sequential(
#             nn.Linear(4096, 256),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, x):
#         x = self.feature_extraction(x)
#         x = x.view(x.size(0), -1)
#         x = self.classification(x)
#         return x
#
# class Discriminator_PAN(nn.Module):
#     def __init__(self):
#         super(Discriminator_PAN, self).__init__()
#         self.in_channels = 1
#         def discriminator_block(in_filters, out_filters, first_block=False):
#             layers = []
#             layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
#             if not first_block:
#                 layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
#             layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         layers = []
#         in_filters = self.in_channels
#         for i, out_filters in enumerate([64,128, 256,512]):
#             layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
#             in_filters = out_filters
#
#         layers += [nn.Conv2d(in_filters, in_filters, 3),
#                    nn.LeakyReLU(0.2),
#                    nn.Conv2d(in_filters, out_filters, 3)]
#
#         self.feature_extraction = nn.Sequential(*layers)
#         self.classification = nn.Sequential(
#             nn.Linear(8192, 100),
#             nn.Linear(100, 1)
#         )
#
#     def forward(self, x):
#         x = self.feature_extraction(x)
#         x = x.view(x.size(0), -1)
#         x = self.classification(x)
#         return x

# if __name__ == "__main__":
#     x = torch.randn(1,31,128,128)
#     PAN = torch.randn(1, 1, 512, 512)
#     model = Discriminator_HS()
#     print(model(x).shape)
