import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
# from .util_relu import conv, predict_flow, deconv, crop_like
from .np_func import conv, predict_flow, deconv, crop_like ,upsample
import numpy as np

class FNSg_np(object):
    def __init__(self):

        self.conv1   = conv() # 7x7 origin
        self.conv1_1 = conv()
        self.conv1_2 = conv(stride=2)
        self.conv2   = conv() # 5x5 origin
        self.conv2_1 = conv(stride=2)
        self.conv3   = conv() # 5x5 origin
        self.conv3_0 = conv(stride=2)
        self.conv3_1 = conv()
        self.conv4   = conv(stride=2)
        self.conv4_1 = conv()
        self.conv5   = conv(stride=2)
        self.conv5_1 = conv()
        self.conv6   = conv(stride=2)
        self.conv6_1 = conv()

        self.deconv5 = deconv()
        self.deconv4 = deconv()
        self.deconv3 = deconv()
        self.deconv2 = deconv()

        self.predict_flow6 = predict_flow()
        self.predict_flow5 = predict_flow()
        self.predict_flow4 = predict_flow()
        self.predict_flow3 = predict_flow()
        self.predict_flow2 = predict_flow()

        self.upsampled_flow6_to_5 = upsample()
        self.upsampled_flow5_to_4 = upsample()
        self.upsampled_flow4_to_3 = upsample()
        self.upsampled_flow3_to_2 = upsample()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         kaiming_normal_(m.weight, 0.1)
        #         if m.bias is not None:
        #             constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_(m.weight, 1)
        #         constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2_1.fw(self.conv2.fw(self.conv1_2.fw(self.conv1_1.fw(self.conv1.fw(x)))))
        out_conv3 = self.conv3_1.fw(self.conv3_0.fw(self.conv3.fw(out_conv2)))
        out_conv4 = self.conv4_1.fw(self.conv4.fw(out_conv3))
        out_conv5 = self.conv5_1.fw(self.conv5.fw(out_conv4))
        out_conv6 = self.conv6_1.fw(self.conv6.fw(out_conv5))

        flow6       = self.predict_flow6.fw(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5.fw(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5.fw(out_conv6), out_conv5)

        concat5 = np.concatenate((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5.fw(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4.fw(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4.fw(concat5), out_conv4)

        concat4 = np.concatenate((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4.fw(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3.fw(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3.fw(concat4), out_conv3)

        concat3 = np.concatenate((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3.fw(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2.fw(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2.fw(concat3), out_conv2)

        concat2 = np.concatenate((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2.fw(concat2)

        return flow2
