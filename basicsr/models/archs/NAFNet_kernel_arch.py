# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

import numpy as np

class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class kernel_extra_Encoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Encoding_Block, self).__init__()
        self.Conv_head = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.Conv_head(input)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        skip = self.act(output)
        output = self.downsample(skip)

        return output, skip


class kernel_extra_conv_mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_mid, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                        )

    def forward(self, input):
        output = self.body(input)
        return output


class kernel_extra_Decoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Decoding_Block, self).__init__()
        self.Conv_t = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.Conv_head = nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.act = nn.ReLU()

    def forward(self, input, skip):
        output = self.Conv_t(input, output_size=[skip.shape[0], skip.shape[1], skip.shape[2], skip.shape[3]])
        output = torch.cat([output, skip], dim=1)
        output = self.Conv_head(output)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        output = self.act(output)

        return output


class kernel_extra_conv_tail(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_tail, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel_mean = self.mean(input)
        # kernel_feature = kernel_mean

        kernel_mean = nn.Softmax2d()(kernel_mean)
        kernel_fea = kernel_mean
        kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)

        # kernel_mean, kernel_var--size:[B, 21, 21]
        return kernel_mean, kernel_fea


class kernel_extra(nn.Module):
    def __init__(self, kernel_size):
        super(kernel_extra, self).__init__()
        self.kernel_size = kernel_size
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 128)
        self.Conv_mid = kernel_extra_conv_mid(128, 256)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(256, 128)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(128, 64)
        self.Conv_tail = kernel_extra_conv_tail(64, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        kernel, kernel_fea = self.Conv_tail(output)

        return kernel, kernel_fea


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=21):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        c_k = c + kernel_size*kernel_size
        self.conv_kernel = nn.Conv2d(in_channels=c_k, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                                     bias=True)

    def forward(self, inp, kernel_fea):
        x = inp

        x = self.norm1(x)

        # kernel_fea [B, 21*21, H, W]
        x = torch.cat([x, kernel_fea], dim=1)
        x = self.conv_kernel(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet_kernel(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], kernel_size=21):
        super().__init__()

        self.kernel_extra = kernel_extra(kernel_size)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            if num==1:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock_kernel(chan, kernel_size=kernel_size) for _ in range(num)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan) for _ in range(num)]
                    )
                )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)


        kernel, kernel_fea = self.kernel_extra(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            if len(encoder) == 1:
                x = encoder[0](x, kernel_fea)
                kernel_fea = F.interpolate(kernel_fea, scale_factor=0.5, mode='bicubic', align_corners=False)
            else:
                x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNet_kernel_Local(Local_Base, NAFNet_kernel):
    def __init__(self, *args, train_size=(1, 3, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet_kernel.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet_kernel_Local(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, kernel_size=19).cuda()

    # inp_shape = (3, 256, 256)

    input = torch.randn((2, 3, 123, 115)).cuda()
    # kernel = torch.randn((2, 21, 21)).cuda()
    with torch.no_grad():
        out = net(input)
    print(out.shape)

    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    #
    # params = float(params[:-3])
    # macs = float(macs[:-4])
    #
    # print(macs, params)
