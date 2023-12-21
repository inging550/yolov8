# 定义检测头以及整体结构
import torch.nn as nn
import torch
from net.block import CBS
from net.backbone import Backbone
from net.neck import Neck
import math

class Yolo8(nn.Module):
  def __init__(self, class_num, phi, pretrain):
    super(Yolo8, self).__init__()
    depth_dict = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
    width_dict = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    deep_width_dict = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
    path = "./model_data/yolov8{}.pt".format(phi)
    dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
    base_c = int(wid_mul * 64)
    base_d = max(round(dep_mul*3), 1)

    self.backbone = Backbone(base_c, base_d, deep_mul)
    self.Neck = Neck(base_c, base_d, deep_mul)
    self.class_num = class_num
    self.stride = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])
    self.detect = Detect(class_num, base_c, deep_mul, self.stride)
    
    if pretrain:  # 是否读入预训练权重
      self.load_pretrain(path)
      self.define_weight()

  def define_weight(self):  # 权重初始化参数
    for i in self.modules():
      if isinstance(i, nn.BatchNorm2d):
        i.eps = 1e-3
        i.momentum = 0.03
      # if isinstance(i, nn.Conv2d):  # 除了输出端，conv都没有bias
      #   nn.init.kaiming_normal_(i.weight, mode="fan_out")

  def load_pretrain(self, path):
    """
    自己定义的模型与官方的名字不一样,所以不能直接torch.load
    """
    pretrain_weight = list(torch.load(path)['model'].state_dict().items()) # 加载官方参数
    self_weight = self.state_dict()
    for index, key in enumerate(self_weight.keys()):
      block_name = key.split('.')[-1]
      if pretrain_weight[index][0].endswith(block_name):
        self_weight[key] = pretrain_weight[index][1]
    self.load_state_dict(self_weight)


  def forward(self, x):
    """
    x [bs 3, 640, 640]
    """
    f1, f2, f3 = self.backbone(x)
    det1, det2, det3 = self.Neck(f1, f2, f3)
    out = self.detect(det1, det2, det3)
    return out
  
  
class Detect(nn.Module):
  def __init__(self, class_num, base_c, deep_mul, stride):
    super(Detect, self).__init__()
    self.reg_max = 16
    self.stride = stride
    c_in = [base_c*4, base_c*8, int(base_c*16*deep_mul)]  # det123的通道数
    self.class_num = class_num
    self.out = [0] * 3
    c2, c3   = max((16, c_in[0] // 4, self.reg_max * 4)), max(c_in[0], class_num) 
    self.reg_conv = nn.ModuleList(nn.Sequential(
                                CBS(i, c2, 3, 1, 1),
                                CBS(c2, c2, 3, 1, 1),
                                nn.Conv2d(c2, 4*self.reg_max, 1, 1)
                                ) for i in c_in)
    self.cls_conv = nn.ModuleList(nn.Sequential(
                                  CBS(i, c3, 3, 1, 1),
                                  CBS(c3, c3, 3, 1, 1),
                                  nn.Conv2d(c3, class_num, 1, 1)
                                  ) for i in c_in)
    self.init_bias()  # 初始化bias
    
  def init_bias(self):
    for a, b, s in zip(self.reg_conv, self.cls_conv, self.stride):
      a[-1].bias.data[:] = 1.0
      b[-1].bias.data[:self.class_num] = math.log(5 / self.class_num / (640/s)**2)


  def forward(self, det1, det2, det3):
    """
    det1 [bs, 64, 80, 80]
    det2 [bs, 128, 40, 40]
    det3 [bs, 256, 20, 20]
    """
    # bs = det1.shape[0]
    # 得到三个det的输出
    for i,det in enumerate([det1, det2, det3]):
      self.out[i] = torch.cat((self.reg_conv[i](det), self.cls_conv[i](det)), 1)
    return self.out
  
