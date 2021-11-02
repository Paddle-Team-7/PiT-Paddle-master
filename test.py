import paddle
import pit

model = pit.pit_ti(pretrained=False)
# print('hello')
model.set_state_dict(paddle.load('./pit_ti_730.pdparams'))
# print('bye')
print(model(paddle.randn([1, 3, 224, 224])))
