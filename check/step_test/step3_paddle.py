import paddle
import pit
from DeiT.losses import DistillationLoss ,SoftTargetCrossEntropyLoss
from DeiT.regnet import build_regnet as build_teacher_model
import numpy as np
from reprod_log import ReprodLogger
# from DeiT.config import get_config, update_config


reprod_logger = ReprodLogger()

criterion = SoftTargetCrossEntropyLoss()

model = pit.pit_ti(pretrained=False)

model.set_state_dict(paddle.load('./pit_ti_730.pdparams'))


fake_data = np.load("fake_data.npy")
fake_label = np.load("fake_label_int.npy")

out = model(paddle.to_tensor(fake_data))
# print(out.detach())
teacher_model = build_teacher_model()
# assert os.path.isfile(config.TRAIN.TEACHER_MODEL + '.pdparams')
# teacher_model_state = paddle.load(config.TRAIN.TEACHER_MODEL + '.pdparams')
# teacher_model.set_dict(teacher_model_state)
# teacher_model.eval()
#print(out.detach())
dis = DistillationLoss(criterion ,teacher_model ,"none" ,0.5 ,1.0)
# print('out-before: ', out.detach())
# print('target-before: ', paddle.to_tensor(fake_label).detach())
loss = dis(paddle.to_tensor(fake_data), out, paddle.to_tensor(fake_label).astype('float32'))

reprod_logger.add("loss", loss.cpu().detach().numpy())
reprod_logger.save("loss_paddle.npy")