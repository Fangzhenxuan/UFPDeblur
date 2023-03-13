
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('PS')


# spatially variant blur
class BatchBlur_SV(nn.Module):
    def __init__(self, l=19, padmode='reflection'):
        super(BatchBlur_SV, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))


    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            pad = pad.view(C * B, 1, H_p, W_p)
            pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]
            kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
            out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
            out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)

            return out
