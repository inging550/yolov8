from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

class CocoData(Dataset):
  """
  从cache加载目标检测数据
  Yolo8输入img为[640, 640]
  """
  def __init__(self, label_cachepath, train):
    super().__init__()
    self.img_path = None
    self.train = train
    if train:
      label_cachepath += '/train2017.cache'
    else:
      label_cachepath += '/val2017.cache'
    self.labels = self.get_labels(label_cachepath)
    self.transform = ToTensor()
    

  def __getitem__(self, index):
    """
    resize + ToTensor + 数据增强
    """
    img_path, label = self.img_path[index], self.labels[index]
    img = Image.open(img_path)
    img, label, scale, x, y = update_img_label(img, label)
    img, label = self.transform(img), self.transform(label).squeeze(0)
    if self.train:
      return img, label
    else:
      return img_path[-16:-4], img, scale, x, y
  
  def __len__(self):
    return len(self.img_path)
  
  def get_labels(self, path):
    cache = load_dataset_cache_file(path)
    if cache['msgs']:
      for i in cache['msgs']:
        print(i)
    # Read cache
    # [cache.pop(k) for k in ('hash', 'version', 'msgs', 'results')] # 是否有必要？
    labels = cache['labels']
    assert len(labels), f'No valid labels found'
    self.img_path = [lb['im_file'] for lb in labels]
    return labels

def load_dataset_cache_file(path):
  import gc
  gc.disable()
  cache = np.load(path, allow_pickle=True).item()  # cache 为字典
  gc.enable()
  return cache

def make_train_data(batch):
  """
  任务: resize + ToTensor 
  """
  imgs, labels = zip(*batch)
  target = []
  for i,label in enumerate(labels):
    obj_num = label.shape[0]
    img_id = torch.full((obj_num, 1), i)
    target.append(torch.cat((img_id, label),1))  
  return torch.stack(imgs, 0), torch.cat(target, 0)

def make_val_data(batch):
  imgname, imgs, scale, x, y = zip(*batch)
  return torch.stack(imgs, 0), imgname, scale, x, y

def update_img_label(img, label):
  width, height = img.width, img.height
  scale = min(640/width, 640/height)
  new_w, new_h = int(width*scale), int(height*scale)
  bboxes = label['bboxes'] # numpy.ndarray
  bboxes[:, [0, 2]] *= (new_w/640)  # 进行rate调整
  bboxes[:, [1, 3]] *= (new_h/640)
  cls = label['cls']
  img = img.resize((new_w, new_h), Image.BICUBIC)
  new_img = Image.new("RGB", (640, 640), (127,127,127))
  x = (640-new_w) / 2
  y = (640-new_h) / 2
  bboxes[:, 0] += (x/640)
  bboxes[:, 1] += (y/640)
  new_img.paste(img, (int(x),int(y)))
  # draw = ImageDraw.Draw(new_img)
  # # new_img.show()
  # for i in range(bboxes.shape[0]):
  #   x1 = int((bboxes[i, 0] - bboxes[i, 2] / 2)*640)
  #   y1 = int((bboxes[i, 1] - bboxes[i, 3] / 2)*640)
  #   x2 = int((bboxes[i, 0] + bboxes[i, 2] / 2)*640)
  #   y2 = int((bboxes[i, 1] + bboxes[i, 3] / 2)*640)
  #   draw.rectangle([x1,y1,x2,y2], outline=(255, 0, 0))
  # new_img.show()
  # print(1)
  return new_img, np.hstack((cls, bboxes)), scale, x, y