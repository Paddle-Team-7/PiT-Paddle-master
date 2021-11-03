import torch
import pit
import numpy as np
import torch.nn as nn
import argparse
import torch.nn.functional as F
from reprod_log import ReprodLogger
# import argparse
from deit.losses import DistillationLoss
#from deit.regnet import build_regnet as build_teacher_model
reprod_logger = ReprodLogger()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
args = parser.parse_args()
# 定义加载模型
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
model = pit.pit_ti(pretrained=False)
model.load_state_dict(torch.load('weights/pit_ti_730.pth',map_location='cpu'))
# 载入数据
fake_data = np.load("fake_data_4.npy")
fake_label = np.load("fake_label_4.npy")
images = torch.tensor(fake_data)
target = torch.tensor(fake_label)
# 定义优化器
model_without_ddp = model
optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=0.0005,
            betas=(0.9, 0.999),
            weight_decay=0.05,
            eps=0.1,
            amsgrad=False
            )
       
model.eval() # 手动处理dropout层
loss_list = []

if args.mixup > 0.:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
else:
    criterion = torch.nn.CrossEntropyLoss()
teacher_model = None
for i in range(5):
    output = model(images)
    dis = DistillationLoss(criterion ,teacher_model ,"none" ,0.5 ,1.0)
    loss = dis(images, output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_list.append(loss)
    print("loss= ",loss)

for idx, loss in enumerate(loss_list):
    reprod_logger.add(f"loss_{idx}", loss.cpu().detach().numpy())
reprod_logger.save('bp_align_torch.npy')




