import time
from tqdm import tqdm
#创建tqdm对象
#迭代
for i in tqdm(range(0,60), ncols=40):
	time.sleep(0.5)