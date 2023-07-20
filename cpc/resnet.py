import torch
import torch.nn as nn


class Basic_block(nn.Module):
    def __init__(self, input_dim, output_dim, downsample = False) -> None:
        super().__init__()
        self.if_down = downsample

        if downsample:
            self.blocks1 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
        else:
            self.blocks1 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )

        self.blocks2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self,x):
        out1 = self.blocks1(x)
        out2 = self.blocks2(out1)

        if self.if_down:
            x = self.downsample(x)
            final_out = out2 + x
        else:
            final_out = out2

        return final_out


class res_for_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(3,64,3,2,1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            Basic_block(64,128,True),
            Basic_block(128,256, True),
            Basic_block(256,512, True)
        )
        self.adptpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512,10)
    def forward(self, x):
        out = self.res(x)
        out = self.adptpool(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        return out

        


        