import torch
from utils.utils import ModelEMA
from net.yolo8 import Yolo8
from torch.utils.data import  DataLoader, RandomSampler, BatchSampler
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from data.dataset import CocoData, make_train_data, make_val_data
from utils.yololoss import Loss
from engine import train_one_epoch, cal_cocomAP
from tqdm import tqdm
import csv, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.cuda import amp

def get_args():
  parser = argparse.ArgumentParser('Yolo8 Object Detector', add_help=False)
  parser.add_argument("--epoch", type=int, default=100)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--nw', type=int, default=6)
  parser.add_argument('--lr0', type=float, default=0.01)
  parser.add_argument('--lrf', type=float, default=0.01)
  parser.add_argument('--momentum', type=float, default=0.937)
  parser.add_argument('--weight_decay', type=float, default=5e-4)
  parser.add_argument('--label_path', type=str, default='/home/zjl/桌面/project/data/cocoData/labels')
  parser.add_argument('--warmup_epoch', type=int, default=3)
  parser.add_argument('--warmup_momentum', type=float, default=0.8)
  args = parser.parse_args()
  return args

def main(args):
  # coco离散类别id与连续类别id的对应关系
  x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
  
  device = torch.device('cuda')
  # 定义数据集
  train_set = CocoData(args.label_path, train=True)
  test_set = CocoData(args.label_path, train=False)
  sampler_train = RandomSampler(train_set)
  batch_train = BatchSampler(sampler_train, batch_size=args.batch_size, drop_last=True)
  train_loader = DataLoader(train_set, batch_sampler=batch_train, num_workers=args.nw, 
                            pin_memory=True, collate_fn=make_train_data)
  test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.nw, pin_memory=True, collate_fn=make_val_data)
  # 定义网络，损失函数以及优化器
  net = Yolo8(80, 'n', True)
  # net.load_state_dict(torch.load("./log/epoch_5 loss_3.994204044342041.pth"))
  net = net.to(device)
  optimizer = torch.optim.SGD(net.parameters(), lr=args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)
  # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4, betas=(0.937, 0.999))
  fn = lambda x : (1 - (1-args.lrf)/args.epoch * x)  # 学习率衰减策略
  schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
  loss_function = Loss(net)
  ema = ModelEMA(net)  
  scaler = amp.GradScaler(enabled=True)
  # # 开始训练
  f = open("train_info.csv", "w", encoding='utf-8')
  csv_writer = csv.writer(f)
  csv_writer.writerow(["epoch", "loss"])
  f.close()
  f = open("test_info.csv", "w", encoding='utf-8')
  csv_writer = csv.writer(f)
  csv_writer.writerow(["epoch", "0.5-0.95", "0.5", "0.75", "s", "m", "l"])
  f.close()
  temp_lr = args.lr0
  with tqdm(range(args.epoch), ncols=80) as tbar:
    for epoch in tbar:
      tbar.set_description("train epoch:%d" % epoch)
      for param in optimizer.param_groups:
        temp_lr = param['lr']
      tbar.set_postfix(lr = temp_lr)
      loss = train_one_epoch(net, optimizer, loss_function, train_loader, device, ema, epoch, args, scaler)
      schedule.step()
      f = open("train_info.csv", "a", encoding='utf-8')
      csv_writer = csv.writer(f)
      csv_writer.writerow([str(epoch), str(loss)])
      f.close()
      if epoch % 5 == 0:
        torch.save(net.state_dict(), './log/epoch:{} loss:{}.pth'.format(epoch, loss))

      with torch.no_grad():
        mAP = cal_cocomAP(net, test_loader, device, 0.7, 0.5, x)
        f = open("test_info.csv", "a", encoding="utf-8")
        csv_writer = csv.writer(f)
        csv_writer.writerow([epoch, str(mAP[0]), str(mAP[1]), str(mAP[2]), str(mAP[3]), str(mAP[4]), str(mAP[5])])
        f.close()
  # with torch.no_grad():
  #   test_one_epoch(net, test_loader, device, "/home/zjl/桌面/project/mAp/Object-Detection-Metrics/detections", 0.85, 0.1)


if __name__ == "__main__":
  args = get_args()
  main(args)
  # coco_true = COCO(annotation_file="/home/zjl/桌面/project/data/cocoData/annotations/instances_val2017.json")
# 载入网络在coco2017验证集上预测的结果
  # coco_pre = coco_true.loadRes('./result.json')

  # coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
  # coco_evaluator.evaluate()
  # coco_evaluator.accumulate()
  # coco_evaluator.summarize()
  # print(1)