# 定义Head端结构
import torch.nn as nn
import torch
from net.block import C2F, CBS

class Neck(nn.Module):
  def __init__(self, base_c, base_d, deep_mul):
    super(Neck, self).__init__()
    self.upsample = nn.Upsample(scale_factor=2)
    self.c2f1 = C2F(int(base_c*16*deep_mul)+base_c*8, base_c*8, base_d, False) # Head端C2F中的Bottlenect都没有残差连接
    self.c2f2 = C2F(base_c*8+base_c*4, base_c*4, base_d, False)
    self.conv1 = CBS(base_c*4, base_c*4, 3, 2, 1)
    self.c2f3 = C2F(base_c*4+base_c*8, base_c*8, base_d, False)
    self.conv2 = CBS(base_c*8, base_c*8, 3, 2, 1)
    self.c2f4 = C2F(int(base_c*16*deep_mul)+base_c*8, int(base_c*16*deep_mul), base_d, False)

  def forward(self, f1, f2, f3):
    """
    f1, f2, f3 来自Backbone
    f1 [bs, 64, 80, 80]
    f2 [bs, 128, 40, 40]
    f3 [bs, 256, 20, 20]
    网络类似N型排列
    """
    x = self.upsample(f3) # [bs, 256, 40, 40]
    x = torch.cat((x, f2), dim=1) # [bs, 256+128=384, 40, 40]
    concat1 = self.c2f1(x) # [bs, 128, 40, 40]
    x = self.upsample(concat1) # [bs, 128, 80, 80]
    x = torch.cat((x, f1), 1) # [bs, 128+64=192, 80, 80]
    det1 = self.c2f2(x) # [bs, 64, 80, 80]
    x = self.conv1(det1) # [bs, 64, 40, 40]
    x = torch.cat((concat1, x), 1) #[bs, 128+64=192, 40, 40]
    det2 = self.c2f3(x) # [bs, 128, 40, 40]
    x = self.conv2(det2) # [bs, 128, 20, 20]
    x = torch.cat((f3, x), 1) #[bs, 256+128=384, 20, 20]
    det3 = self.c2f4(x) # [bs, 256, 20, 20]
    return det1, det2, det3