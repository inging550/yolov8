import torch
from net.yolo8 import Yolo8
net = Yolo8(80, 'l', True)
# print(net)
# net.load_state_dict(torch.load("./yolo8.pth"))
# print(len(net.state_dict().items()))
for i in net.named_parameters():
  print(i[0], i[1].shape, sep="  ")