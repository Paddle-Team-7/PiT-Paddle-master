import torch
import numpy as np
from pit import pit_ti
from reprod_log import ReprodLogger
reprod_logger = ReprodLogger()
model = pit_ti(pretrained=False)
model.load_state_dict(torch.load('weights/pit_ti_730.pth',map_location='cpu'))
model.eval()
fake_data = np.load("fake_data.npy")
fake_data=torch.tensor(fake_data)
out=model(fake_data)
out=out.cpu().detach().numpy()
print(out)
print(type(out))
reprod_logger.add("out", out.cpu().detach().numpy())
reprod_logger.save("forward_torch.npy")