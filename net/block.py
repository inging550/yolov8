# 定义一些子模块
import torch.nn as nn
import torch

class CBS(nn.Module):
  def __init__(self, c_in, c_out, k_size, stride, padding):
    super(CBS, self).__init__()
    self.conv = nn.Conv2d(c_in, c_out, k_size, stride, padding, bias=False)
    self.bn = nn.BatchNorm2d(c_out)
    self.silu = nn.SiLU(inplace=True)

  def forward(self, x):
    """ 
    Conv2d + BatchNorm + SiLU
    """
    return self.silu(self.bn(self.conv(x)))
  
class C2F(nn.Module):
  def __init__(self, c_in, c_out, n, shortcut):
    """
    n: Bottleneck的数量
    """
    super(C2F, self).__init__()
    self.conv1 = CBS(c_in, c_out, 1, 1, 0)
    self.conv2 = CBS(int(c_out*0.5*(n+2)), c_out, 1, 1, 0)
    self.module = nn.ModuleList([Bottleneck(shortcut, c_out//2, c_out//2, e=1) for _ in range(n)])

  def forward(self, x:torch.Tensor):
    """
    x [batch_size, c_in, h, w]
    return [batch_size, c_out*0.5*(n+2), h, w]
    """
    x = self.conv1(x)  # shape不变
    x1, x2 = x.chunk(2, dim=1)  # 将channel切成两块
    output = [x1, x2]
    for layer in self.module:
      output.append(layer(output[-1]))
    return self.conv2(torch.cat(output, 1)) 
  
class Bottleneck(nn.Module):
  def __init__(self, shortcut:bool, c_in:int, c_out:int, e=0.5):
    super(Bottleneck, self).__init__()
    self.shortcut = shortcut and c_in==c_out
    c_hid = int(c_out * e)
    self.conv1 = CBS(c_in, c_hid, 3, 1, 1)  # 这里有错误应为CBS
    self.conv2 = CBS(c_hid, c_out, 3, 1, 1)

  def forward(self, x):
    """
    shortcut: 是否进行残差连接
    两层卷积(然后进行残差连接)
    """
    out = self.conv2(self.conv1(x))
    out = x+out if self.shortcut else out
    return out
  
class SPFF(nn.Module):
  def __init__(self, channel):
    super(SPFF, self).__init__()
    self.conv1 = CBS(channel, channel//2, 1, 1, 0)
    self.conv2 = CBS(channel*2, channel, 1, 1, 0)
    self.maxpool = nn.MaxPool2d(5, 1, 5//2)

  def forward(self, x):
    """
    x [batch_size, 256, 20, 20]
    conv1 -> 三层maxpool -> concat -> conv2 -> return 
    return [batch_size, 256, 20, 20]
    """
    x = self.conv1(x)
    y1 = self.maxpool(x)
    y2 = self.maxpool(y1)
    return self.conv2(torch.cat((x,y1,y2,self.maxpool(y2)), 1))