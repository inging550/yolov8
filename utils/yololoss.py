from utils.utils import make_anchors, preprocess, bbox_decode, bbox_iou, bbox2dist
from utils.boxassign import TaskAlignedAssigner
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
  def __init__(self, model):
    self.class_num = model.class_num
    self.no = self.class_num + 4*16
    self.stride = model.stride
    self.assigner = TaskAlignedAssigner(topk=10,
                                        num_classes=self.class_num,
                                        alpha=0.5,
                                        beta=6.0,
                                        roll_out_thr=64)
    self.bce = nn.BCEWithLogitsLoss(reduction='none')
    device = next(model.parameters()).device 
    self.bbox_loss = BboxLoss(15, use_dfl=True).to(device)
    # self.dfl_decode = DFL(16)

  def __call__(self, preds, target):
    device = preds[0].device
    bs = preds[0].shape[0]
    loss = torch.zeros(3, device=device) # 初始化loss

    # 对preds进行操作
    # pre_cls [bs, class_num, 8400]
    # pre_distri [bs, 64, 8400]
    pred_distri, pred_cls = torch.cat(([i.view(bs, self.no, -1) for i in preds]), 2).split([64, self.class_num], 1)
    pred_cls = pred_cls.permute(0,2,1).contiguous()
    pred_distri = pred_distri.permute(0,2,1).contiguous()
    dtype = pred_cls.dtype
    img_size = torch.tensor(preds[0].shape[-2:], device=device, dtype=dtype) * self.stride[0]
    anchor_point, stride_tensor = make_anchors(preds, self.stride)
    pred_bbox = bbox_decode(anchor_point, pred_distri) # [bs, 8400, 4] xyxy
    
    # 对target进行操作 range = [0, img_size]
    target = preprocess(target, bs, img_size) 
    tar_cls, tar_bbox = target.split([1, 4], 2)
    mask_tar = tar_bbox.sum(2, keepdim=True).gt_(0) # [bs, 17, 1] 那些不是填充的

    # 开始配对
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_cls.detach().sigmoid(), (pred_bbox.detach() * stride_tensor).type(tar_bbox.dtype),
            anchor_point * stride_tensor, tar_cls, tar_bbox, mask_tar
        )
    
    target_scores_sum = max(target_scores.sum(), 1)

    # 计算loss
    loss[1] = self.bce(pred_cls, target_scores.to(dtype)).sum() / target_scores_sum
    if fg_mask.sum():
      target_bboxes /= stride_tensor
      loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bbox, anchor_point, target_bboxes, 
                                        target_scores, target_scores_sum, fg_mask)

    loss[0] *= 7.5
    loss[1] *= 0.5
    loss[2] *= 1.5
    return loss.sum()*bs, loss.detach()
  
class BboxLoss(nn.Module):
  def __init__(self, reg_max=16, use_dfl=False):
    super().__init__()
    self.reg_max = reg_max
    self.use_dfl = use_dfl

  def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
    # 计算IOU损失
    # weight代表损失中标签应该有的置信度，0最小，1最大s
    # weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    # 计算预测框和真实框的重合程度
    iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    # 然后1-重合程度，乘上应该有的置信度，求和后求平均。
    loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum

    # 计算DFL损失
    if self.use_dfl:
      target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
      loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
      loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
      loss_dfl = torch.tensor(0.0).to(pred_dist.device)

    return loss_iou, loss_dfl

  @staticmethod
  def _df_loss(pred_dist, target):
    # Return sum of left and right DFL losses
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    tl = target.long()  # target left
    tr = tl + 1  # target right
    wl = tr - target  # weight left
    wr = 1 - wl  # weight right
    # 一个点一般不会处于anchor点上，一般是xx.xx。如果要用DFL的话，不可能直接一个cross_entropy就能拟合
    # 所以把它认为是相对于xx.xx左上角锚点与右下角锚点的距离 如果距离右下角锚点距离小，wl就小，左上角损失就小
    #                                                   如果距离左上角锚点距离小，wr就小，右下角损失就小
    return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
            F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)
