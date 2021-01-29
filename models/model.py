import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *


class My_Model(nn.Module):
    def __init__(self, bitW, bitA, classes=3):
        super(My_Model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            Act_Q(bitA=bitA),
            
            Conv2d_Q(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, bitW=bitW),
            Act_Q(bitA=bitA),
            
            Conv2d_Q(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False, bitW=bitW),
            Act_Q(bitA=bitA),

            Conv2d_Q(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, bitW=bitW),
            Act_Q(bitA=bitA),
            
            Conv2d_Q(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False, bitW=bitW),
            Act_Q(bitA=bitA),

            Conv2d_Q(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False, bitW=bitW),
            Act_Q(bitA=bitA),
        )
        #self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*16, out_features=classes, bias=False),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x





