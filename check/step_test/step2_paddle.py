import paddle
import pit
import numpy as np
from reprod_log import ReprodLogger


import paddle.nn.functional as F

reprod_logger = ReprodLogger()

model = pit.pit_ti(pretrained=False)
model.set_state_dict(paddle.load('./pit_ti_730.pdparams'))

model.eval()

fake_data = np.load("fake_data_150.npy")
fake_label = np.load("fake_label_150.npy")
print(paddle.to_tensor(fake_label).detach())
# fake_label = paddle.full([10],50).astype('int64')
# print(paddle.to_tensor(fake_label).detach())


output = model(paddle.to_tensor(fake_data))

# loss = DistillationLoss(fake_data,output,fake_label,0.5, 1.0)
# print(output.detach())

pred = F.softmax(output)
# print(output.detach())

acc1 = paddle.metric.accuracy(output, paddle.to_tensor(fake_label).unsqueeze(1))
acc5 = paddle.metric.accuracy(output, paddle.to_tensor(fake_label).unsqueeze(1), k=5)

print("acc1",acc1)
print("acc5",acc5)

reprod_logger.add("acc1", acc1.cpu().detach().numpy())
reprod_logger.add("acc5", acc5.cpu().detach().numpy())
reprod_logger.save('metric_paddle.npy')


