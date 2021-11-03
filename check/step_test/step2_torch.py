import torch
import pit
import numpy as np
#from timm.utils import accuracy
from reprod_log import ReprodLogger
reprod_logger = ReprodLogger()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) / batch_size for k in topk]

model = pit.pit_ti(pretrained=False)
model.load_state_dict(torch.load('weights/pit_ti_730.pth',map_location='cpu'))
model.eval()
fake_data = np.load("fake_data_150.npy")
fake_label = np.load("fake_label_150.npy")
out = model(torch.tensor(fake_data))
acc1, acc5 = accuracy(out, torch.tensor(fake_label), topk=(1, 5))
acc1=acc1.cpu().detach().numpy()
acc5=acc5.cpu().detach().numpy()
reprod_logger.add("acc1", acc1)
reprod_logger.add("acc5", acc5)
reprod_logger.save("metric_torch_acc.npy")