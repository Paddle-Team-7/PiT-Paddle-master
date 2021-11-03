import torch
import pit
from deit.losses import DistillationLoss
import numpy as np
from reprod_log import ReprodLogger
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
args = parser.parse_args()
model = pit.pit_ti(pretrained=False)
model.load_state_dict(torch.load('weights/pit_ti_730.pth',map_location='cpu'))
fake_data = np.load("fake_data.npy")
fake_label = np.load("fake_label_int.npy")
teacher_model = None
criterion = LabelSmoothingCrossEntropy()
if args.mixup > 0.:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
elif args.smoothing:
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
else:
    criterion = torch.nn.CrossEntropyLoss()
dis = DistillationLoss(
    criterion, teacher_model, None, 0.5, 1.0
)
out = model(torch.tensor(fake_data))
loss = dis(torch.tensor(fake_data),out,torch.tensor(fake_label))
reprod_logger = ReprodLogger()
reprod_logger.add("loss", loss.cpu().detach().numpy())
reprod_logger.save("loss_torch.npy")