import paddle
import pit
import numpy as np
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()


model = pit.pit_ti(pretrained=False)

# print('hello')
model.set_state_dict(paddle.load('./pit_ti_730.pdparams'))
# print('bye')
model.eval()
fake_data = np.load("fake_data.npy")
#fake_label = np.load()

out = model(paddle.to_tensor(fake_data))

reprod_logger.add("out", out.cpu().detach().numpy())
reprod_logger.save('forward_paddle.npy')

