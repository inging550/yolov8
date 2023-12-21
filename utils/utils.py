import torch.nn as nn
import torch
import math
from copy import deepcopy

# 坐标解码  4*16 -> 4*1
class DFL(nn.Module):
  def __init__(self, c1=16):
      super(DFL, self).__init__()
      self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)  # 参数不会更新
      x           = torch.arange(c1, dtype=torch.float)
      self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
      self.c1     = c1

  def forward(self, x):
      b, c, a = x.shape
      # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
      return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
  
def bbox_decode(anchor_points, pre_distri, if_xywh=False):
  """
  anchor_points [8400, 2]
  pre_distri [bs, 8400, 64]
  """
  bs, anchor, _ = pre_distri.shape
  out = pre_distri.view(bs, anchor, 4, 16).softmax(3).matmul(torch.arange(16, device=pre_distri.device, dtype=pre_distri.dtype))
  # out [bs, 8400, 4]
  return dist2bbox(out, anchor_points, if_xywh=if_xywh)

def dist2bbox(x, anchor_points, if_xywh):
  """
  x [bs, 8400, 4]
  anchor [8400, 2]
  return [bs, 8400, 4] -> xyxy
  """
  min_offset, max_offset = x.chunk(2, -1)
  x1y1 = anchor_points - min_offset
  x2y2 = anchor_points + max_offset
  if if_xywh: # 左上角的坐标
    wh = x2y2 - x1y1
    c_xy = x1y1
    return torch.cat((c_xy, wh), -1)
  return torch.cat((x1y1, x2y2), -1)

def make_anchors(preds, strides, offset=0.5):
  """
  preds[0] [bs, 84, 80, 80]
  preds[1] [bs, 84, 40, 40]
  preds[2] [bs, 84, 20, 20]
  strides: 下采样倍数  [8, 16, 32]
  return : 各点的坐标
  """
  anchor_points , stride_tensor = [], []
  device, dtype = preds[0].device, preds[0].dtype
  for i,stride in enumerate(strides):
    _,_,h,w = preds[i].shape
    xlist = torch.arange(offset, w, dtype=dtype, device=device)
    ylist = torch.arange(offset, h, dtype=dtype, device=device)
    y, x = torch.meshgrid(ylist, xlist, indexing = 'ij')
    anchor_points.append(torch.stack((x,y), -1).view(-1, 2))
    stride_tensor.append(torch.full((h*w, 1), stride, device=device, dtype=dtype))
  anchor_points = torch.cat(anchor_points)
  stride_tensor = torch.cat(stride_tensor)
  return anchor_points, stride_tensor

def preprocess(target, bs, img_size):
  
  """
  target [num_target, 6]  
    6  = 第i张图片 + class + x + y + w + h
    xywh -> 归一化的坐标
  return [bs, max_obj, 5]
    max_obj -> bs张图片中,单张图片所含最大obj数量
   """
  
  if target.shape[0] == 0:
     out = torch.zeros((bs, 0, 5), device=target.device)
  else:
    i = target[:, 0]
    _, count = i.unique(return_counts=True)
    max_obj = count.int().max()
    out = torch.zeros((bs, max_obj, 5), device=target.device)
    for j in range(bs):
      mask_i = (i == j)
      n = mask_i.sum()
      if n:
        out[j, :n] = target[mask_i, 1:]
    out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(img_size[[1, 0, 1, 0]]))
  return out

def xywh2xyxy(x):  # 这里对于coco数据集可能不兼容
  """
  x [bs, obj, 4]
  """
  y = torch.empty_like(x)
  dw = x[..., 2] / 2
  dh = x[..., 3] / 2
  y[..., 0] = x[..., 0] - dw  # top left x
  y[..., 1] = x[..., 1] - dh  # top left y
  y[..., 2] = x[..., 0] + dw  # bottom right x
  y[..., 3] = x[..., 1] + dh  # bottom right y
  return y

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
  # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

  # Get the coordinates of bounding boxes
  if xywh:  # transform from xywh to xyxy
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
  else:  # x1, y1, x2, y2 = box1
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

  # Intersection area
  inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
          (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

  # Union Area
  union = w1 * h1 + w2 * h2 - inter + eps

  # IoU
  iou = inter / union
  if CIoU or DIoU or GIoU:
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
      c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
      rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
      if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
          alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
      return iou - rho2 / c2  # DIoU
    c_area = cw * ch + eps  # convex area
    return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
  return iou  # IoU

def bbox2dist(anchor_points, bbox, reg_max):
  """Transform bbox(xyxy) to dist(ltrb)."""
  x1y1, x2y2 = torch.split(bbox, 2, -1)
  return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

class ModelEMA:
  """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
  Keeps a moving average of everything in the model state_dict (parameters and buffers)
  For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
  To disable EMA set the `enabled` attribute to `False`.
  """

  def __init__(self, model, decay=0.9999, tau=2000, updates=0):
    """Create EMA."""
    self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
    self.updates = updates  # number of EMA updates
    self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
    for p in self.ema.parameters():
      p.requires_grad_(False)
    self.enabled = True

  def update(self, model):
    """Update EMA parameters."""
    if self.enabled:
      self.updates += 1
      d = self.decay(self.updates)

      msd = de_parallel(model).state_dict()  # model state_dict
      for k, v in self.ema.state_dict().items():
        if v.dtype.is_floating_point:  # true for FP16 and FP32
          v *= d
          v += (1 - d) * msd[k].detach()
          # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

  def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
    """Updates attributes and saves stripped model with optimizer removed."""
    if self.enabled:
      copy_attr(self.ema, model, include, exclude)


def copy_attr(a, b, include=(), exclude=()):
  """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
  for k, v in b.__dict__.items():
    if (len(include) and k not in include) or k.startswith('_') or k in exclude:
      continue
    else:
      setattr(a, k, v)

def de_parallel(model):
  """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
  return model.module if is_parallel(model) else model

def is_parallel(model):
  """Returns True if model is of type DP or DDP."""
  return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
