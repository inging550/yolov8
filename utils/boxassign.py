import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import bbox_iou

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
  """select the positive anchor center in gt

  Args:
      xy_centers (Tensor): shape(h*w, 4)
      gt_bboxes (Tensor): shape(b, n_boxes, 4)
  Return:
      (Tensor): shape(b, n_boxes, h*w)
  """
  n_anchors       = xy_centers.shape[0]
  bs, n_boxes, _  = gt_bboxes.shape
  # 计算每个真实框距离每个anchors锚点的左上右下的距离，然后求min
  # 保证真实框在锚点附近，包围锚点
  if roll_out:
    bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
    for b in range(bs):
      lt, rb          = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
      bbox_deltas[b]  = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                  dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
    return bbox_deltas
  else:
    # 真实框的坐上右下left-top, right-bottom 
    lt, rb      = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  
    # 真实框距离每个anchors锚点的左上右下的距离
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
  """if an anchor box is assigned to multiple gts,
      the one with the highest iou will be selected.

  Args:
      mask_pos (Tensor): shape(b, n_max_boxes, h*w)
      overlaps (Tensor): shape(b, n_max_boxes, h*w)
  Return:
      target_gt_idx (Tensor): shape(b, h*w)
      fg_mask (Tensor): shape(b, h*w)
      mask_pos (Tensor): shape(b, n_max_boxes, h*w)
  """
  # b, n_max_boxes, 8400 -> b, 8400
  fg_mask = mask_pos.sum(-2)
  # 如果有一个anchor被指派去预测多个真实框
  if fg_mask.max() > 1:  
    # b, n_max_boxes, 8400
    mask_multi_gts      = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  
    # 如果有一个anchor被指派去预测多个真实框，首先计算这个anchor最重合的真实框
    # 然后做一个onehot
    # b, 8400
    max_overlaps_idx    = overlaps.argmax(1)  
    # b, 8400, n_max_boxes
    is_max_overlaps     = F.one_hot(max_overlaps_idx, n_max_boxes)  
    # b, n_max_boxes, 8400
    is_max_overlaps     = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  
    # b, n_max_boxes, 8400
    mask_pos            = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) 
    fg_mask             = mask_pos.sum(-2)
  # 找到每个anchor符合哪个gt
  target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
  return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
  """
  进行双向选择
  1、计算所有预测与真实框的匹配度  align_metric=s^a + ciou^b -> s是预测类别分值 a,b为超参数
  1、根据预测框选择与target iou最大的topk
  2、可能有些预测同属多个target, 根据最大的ciou值分配
  """
  def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
    super().__init__()
    self.topk           = topk
    self.num_classes    = num_classes
    self.bg_idx         = num_classes
    self.alpha          = alpha
    self.beta           = beta
    self.eps            = eps
    # roll_out_thr为64
    self.roll_out_thr   = roll_out_thr

  @torch.no_grad()
  def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    """This code referenced to
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

    Args:
        pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
        pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
        anc_points (Tensor) : shape(num_total_anchors, 2)
        gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
        gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
        mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
    Returns:
        target_labels (Tensor)  : shape(bs, num_total_anchors)
        target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
        target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
        fg_mask (Tensor)        : shape(bs, num_total_anchors)
    """
    # 获得batch_size 
    self.bs             = pd_scores.size(0)
    # 获得真实框中的最大框数量
    self.n_max_boxes    = gt_bboxes.size(1)
    # 如果self.n_max_boxes大于self.roll_out_thr则roll_out
    self.roll_out       = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

    if self.n_max_boxes == 0:
        device = gt_bboxes.device
        return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device))

    # b, max_num_obj, 8400
    # mask_pos      满足在真实框内、是真实框topk最重合的正样本、满足mask_gt的锚点
    # align_metric  某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
    # overlaps      所有真实框和锚点的重合程度
    mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

    # target_gt_idx     b, 8400     每个anchor符合哪个gt
    # fg_mask           b, 8400     每个anchor是否有符合的gt
    # mask_pos          b, max_num_obj, 8400    one_hot后的target_gt_idx
    target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

    # 指定目标到对应的anchor点上
    # b, 8400
    # b, 8400, 4
    # b, 8400, 80
    target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

    # 乘上mask_pos，把不满足真实框满足的锚点的都置0
    align_metric        *= mask_pos
    # 每个真实框对应的最大得分
    # b, max_num_obj
    pos_align_metrics   = align_metric.amax(axis=-1, keepdim=True) 
    # 每个真实框对应的最大重合度
    # b, max_num_obj
    pos_overlaps        = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
    # 把每个真实框和先验点的得分乘上最大重合程度，再除上最大得分
    norm_align_metric   = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
    # target_scores作为正则的标签
    target_scores       = target_scores * norm_align_metric

    return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
  
  def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
    # pd_scores bs, num_total_anchors, num_classes
    # pd_bboxes bs, num_total_anchors, 4
    # gt_labels bs, n_max_boxes, 1
    # gt_bboxes bs, n_max_boxes, 4
    # 
    # align_metric是一个算出来的代价值，某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
    # overlaps是某个先验点与真实框的重合程度
    # align_metric, overlaps    bs, max_num_obj, 8400
    align_metric, overlaps  = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
    
    # 正样本锚点需要同时满足：
    # 1、在真实框内
    # 2、是真实框topk最重合的正样本
    # 3、满足mask_gt
    
    # get in_gts mask           b, max_num_obj, 8400
    # 判断先验点是否在真实框内
    mask_in_gts             = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
    # get topk_metric mask      b, max_num_obj, 8400
    # 判断锚点是否在真实框的topk中
    mask_topk               = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
    # merge all mask to a final mask, b, max_num_obj, h*w
    # 真实框存在，非padding
    mask_pos                = mask_topk * mask_in_gts * mask_gt

    return mask_pos, align_metric, overlaps
  
  def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
    if self.roll_out:
      align_metric    = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
      overlaps        = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
      ind_0           = torch.empty(self.n_max_boxes, dtype=torch.long)
      for b in range(self.bs):
        ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
        # 获得属于这个类别的得分
        # bs, max_num_obj, 8400
        bbox_scores     = pd_scores[ind_0, :, ind_2]  
        # 计算真实框和预测框的ciou
        # bs, max_num_obj, 8400
        overlaps[b]     = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True).squeeze(2).clamp(0)
        align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
    else:
      # 2, b, max_num_obj
      ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)       
      # b, max_num_obj  
      # [0]代表第几个图片的
      ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  
      # [1]真是标签是什么
      ind[1] = gt_labels.long().squeeze(-1) 
      # 获得属于这个类别的得分
      # 取出某个先验点属于某个类的概率
      # b, max_num_obj, 8400
      bbox_scores = pd_scores[ind[0], :, ind[1]]  

      # 计算真实框和预测框的ciou
      # bs, max_num_obj, 8400
      overlaps        = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
      align_metric    = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps
  
  def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
    """
    Args:
        metrics     : (b, max_num_obj, h*w).
        topk_mask   : (b, max_num_obj, topk) or None
    """
    # 8400
    num_anchors             = metrics.shape[-1] 
    # b, max_num_obj, topk
    topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
    if topk_mask is None:
      topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
    # b, max_num_obj, topk
    topk_idxs[~topk_mask] = 0
    # b, max_num_obj, topk, 8400 -> b, max_num_obj, 8400
    # 这一步得到的is_in_topk为b, max_num_obj, 8400
    # 代表每个真实框对应的top k个先验点
    if self.roll_out:
      is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
      for b in range(len(topk_idxs)):
          is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
    else:
      is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
    # 判断锚点是否在真实框的topk中
    is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
    return is_in_topk.to(metrics.dtype)
  
  def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
    """
    Args:
        gt_labels       : (b, max_num_obj, 1)
        gt_bboxes       : (b, max_num_obj, 4)
        target_gt_idx   : (b, h*w)
        fg_mask         : (b, h*w)
    """

    # 用于读取真实框标签, (b, 1)
    batch_ind       = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
    # b, h*w    获得gt_labels，gt_bboxes在flatten后的序号
    target_gt_idx   = target_gt_idx + batch_ind * self.n_max_boxes
    # b, h*w    用于flatten后读取标签
    target_labels   = gt_labels.long().flatten()[target_gt_idx]
    # b, h*w, 4 用于flatten后读取box
    target_bboxes   = gt_bboxes.view(-1, 4)[target_gt_idx]
    
    # assigned target scores
    target_labels.clamp(0)
    # 进行one_hot映射到训练需要的形式。
    target_scores   = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
    fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
    target_scores   = torch.where(fg_scores_mask > 0, target_scores, 0)

    return target_labels, target_bboxes, target_scores
