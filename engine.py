import torch
from tqdm import tqdm
from pathlib import Path
from utils.utils import make_anchors, bbox_decode
from torchvision.ops import nms
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json



def train_one_epoch(net, optimizer, loss_function, data_loader, device, ema, epoch, args, scaler):
  interval, i = 300, 0
  bbox_loss, cls_loss, dfl_loss = 0, 0, 0
  num_batch = len(data_loader)
  warmup_data = num_batch*args.warmup_epoch
  net.train()
  with tqdm(data_loader, ncols=50) as tbar:
    for (img, target) in tbar:
      tbar.set_description("Epoch %d" % epoch)
       # 前几个epoch要对学习率进行预热
      if epoch < args.warmup_epoch:
        for x in optimizer.param_groups:
          x['lr'] = np.interp(i+epoch*num_batch, [0, warmup_data], [0, args.lr0] )
          if 'momentum' in x:
            x['momentum'] = np.interp(i+epoch*num_batch, [0, warmup_data], [args.warmup_momentum, args.momentum])
      img, target = img.to(device), target.to(device)
      i += 1
      with torch.cuda.amp.autocast(True):
        out = net(img)
        tloss, loss= loss_function(out, target)
        bbox_loss += loss[0]
        cls_loss += loss[1]
        dfl_loss += loss[2]
      if i % interval == 0:
        tbar.write("bbox_loss:{}, cls_loss:{}, dfl_loss:{}".format(bbox_loss/interval, cls_loss/interval, dfl_loss/interval))
        bbox_loss, cls_loss, dfl_loss = 0, 0, 0
      scaler.scale(tloss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # clip gradients 梯度裁减
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()
      ema.update(net)
  return loss.sum().item()

def cal_class_map(net, data_loader, device, out_path, conf_thread, iou_threshold, epoch):
  p = Path(out_path)
  flag = True
  with tqdm(data_loader, ncols=80) as tbar:
    for (img, imgname) in tbar:
      img = img.to(device)
      out = net(img)
      # 开始解码输出
      if flag:
        anchor_points, stride_tensor = make_anchors(out, net.stride)
        flag = False
      bs = img.shape[0]
      no = out[0].shape[1]
      pre_distri, pre_cls = torch.cat([i.view(bs, no, -1) for i in out], 2).split([64, net.class_num], 1)
      pre_distri = pre_distri.permute(0, 2, 1) # [bs, 8400, 64]
      pre_cls = pre_cls.permute(0, 2, 1).softmax(2) # [bs, 8400, 80]
      # 将 dbox变换到 xyxy格式
      pre_bboxes = bbox_decode(anchor_points, pre_distri)
      pre_bboxes *= stride_tensor # 放大到640，640尺度
      # 进行nms
      for j,(bbox, cls) in enumerate(zip(pre_bboxes, pre_cls)):
        file_name = p / (imgname[i]+".txt")
        cls_conf, cls_pred = torch.max(cls, 1, keepdim=True) # [8400, 1]
        
        conf_mask = (cls_conf[:, 0] >= conf_thread).squeeze()
        bbox = bbox[conf_mask]
        cls_conf = cls_conf[conf_mask]
        cls_pred = cls_pred[conf_mask]
        if not cls.shape[0]:
          f = open(file_name , 'w')
          f.close()
          continue
        detections = torch.cat([bbox, cls_conf.float(), cls_pred.float()], 1)
        unique_labels = detections[:, -1].unique()
        f = open(file_name, 'w')
        for c in unique_labels:
          class_mask = detections[:, -1]==c
          detection_i = detections[class_mask]
          keep = nms(boxes=detection_i[:,:4], scores=detection_i[:, 4], iou_threshold=iou_threshold)
          result = detection_i[keep]  # [n, 6]  x1 y1 x2 y2 conf cls
          for i in range(result.shape[0]):
            out_c = str(int(result[i, -1].item()))
            conf = str(result[i, 4].item())
            x1 = str(int(result[i, 0].item()))
            y1 = str(int(result[i, 1].item()))
            x2 = str(int(result[i, 2].item()))
            y2 = str(int(result[i, 3].item()))
            f.write((out_c + " " + conf + " " + x1 + " " + y1 + " " + x2 + " " + y2 + "\n"))
        f.close()
  return 

def cal_cocomAP(net, dataloader, device, conf_thread, iou_threshold, cls_id):
  results = [] # 保存验证集所有的识别信息
  flag = True
  for img, imgid, scale, x_offset, y_offset in tqdm(dataloader):
    img = img.to(device)
    out = net(img)
    # 最终需要image_id + category_id(91) + bbox + score
    if flag:
      anchor_points, stride_tensor = make_anchors(out, net.stride)
      flag = False
    bs = img.shape[0]
    no = out[0].shape[1]
    pre_distri, pre_cls = torch.cat([i.view(bs, no, -1) for i in out], 2).split([64, net.class_num], 1)
    pre_distri = pre_distri.permute(0, 2, 1) # [bs, 8400, 64]
    pre_cls = pre_cls.permute(0, 2, 1).softmax(2) # [bs, 8400, 80]
    # dbox 切换到xyxy格式
    pre_bboxes = bbox_decode(anchor_points, pre_distri) 
    pre_bboxes *= stride_tensor # 放大到640，640尺度
    # 进行nms, 遍历batch_size
    for i,(bbox, cls) in enumerate(zip(pre_bboxes, pre_cls)):
      cls_conf, cls_pred = torch.max(cls, 1, keepdim=True) # [8400, 1]
      conf_mask = (cls_conf[:, 0] >= conf_thread).squeeze()
      bbox = bbox[conf_mask]
      cls_conf = cls_conf[conf_mask]
      cls_pred = cls_pred[conf_mask]
      if not cls.shape[0]:
        continue
      detections = torch.cat([bbox, cls_conf.float(), cls_pred.float()], 1)
      unique_labels = detections[:, -1].unique()
      for c in unique_labels:
        class_mask = detections[:, -1]==c
        detection_i = detections[class_mask]
        keep = nms(boxes=detection_i[:,:4], scores=detection_i[:, 4], iou_threshold=iou_threshold)
        result = detection_i[keep]  # [n, 6]  x1 y1 x2 y2 conf cls
        result[:, :4] /= scale[i]
        result[:, [0, 2]] -= x_offset[i]
        result[:, [1, 3]] -= y_offset[i]
        for j in range(result.shape[0]):
          out_c = cls_id[int(result[j, -1].item())]
          conf = result[j, 4].item()
          x = result[j, 0].item()
          y = result[j, 1].item()
          w = result[j, 2].item() - x
          h = result[j, 3].item() - y
          results.append(dict(
            image_id = int(imgid[i]),
            category_id = out_c,
            bbox = [x, y, w, h],
            score = conf))
  with open('result.json', 'w') as f: # 保存为json文件
    json.dump(results, f)
    coco_true = COCO(annotation_file="/home/zjl/桌面/project/data/cocoData/annotations/instances_val2017.json")
  # 载入网络在coco2017验证集上预测的结果
  coco_pre = coco_true.loadRes('./result.json')

  coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
  coco_evaluator.evaluate()
  coco_evaluator.accumulate()
  coco_evaluator.summarize()
  return coco_evaluator.stats[:6]
