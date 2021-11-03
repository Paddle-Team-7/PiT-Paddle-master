import paddle
import pit
import numpy as np
from reprod_log import ReprodLogger

# import argparse
from DeiT.losses import DistillationLoss
from DeiT.regnet import build_regnet as build_teacher_model
from DeiT.losses import DistillationLoss ,SoftTargetCrossEntropyLoss

reprod_logger = ReprodLogger()

# 定义加载模型

model = pit.pit_ti(pretrained=False)
model.set_state_dict(paddle.load('./pit_ti_730.pdparams'))

# 载入数据
fake_data = np.load("fake_data.npy")
fake_label = np.load("fake_label.npy")

images = paddle.to_tensor(fake_data)
target = paddle.to_tensor(fake_label)

# 定义优化器
model_without_ddp = model

#optimizer = create_optimizer(args, model_without_ddp)

optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=0.0005,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.05,
            epsilon=0.1,
            grad_clip=None,
            )
       
model.eval() # 手动处理dropout层

loss_list = []

criterion = SoftTargetCrossEntropyLoss()
teacher_model = build_teacher_model()
# print(teacher_model)

for i in range(5):

    output = model(images)
    dis = DistillationLoss(criterion ,teacher_model ,"none" ,0.5 ,1.0)
    # print('out-before: ', out.detach())
    # print('target-before: ', paddle.to_tensor(fake_label).detach())
    loss = dis(images, output, target.astype('float64'))
    # loss = DistillationLoss(fake_data,output,target)

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    loss_list.append(loss.detach())

    print("loss= ",loss.detach())
    reprod_logger.add("loss_{i}", loss.cpu().detach().numpy())
reprod_logger.save('bp_align_paddle.npy')




