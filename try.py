import numpy as np

import paddle
import paddle.static as static
import paddle.nn.functional as F

paddle.enable_static()
data = static.data(name="input", shape=[-1, 32, 32], dtype="float32")
label = static.data(name="label", shape=[-1,1], dtype="int")
fc_out = static.nn.fc(x=data, size=10)
predict = F.softmax(x=fc_out)
result = static.accuracy(input=predict, label=label, k=5)

place = paddle.CPUPlace()
exe = static.Executor(place)

exe.run(static.default_startup_program())
x = np.random.rand(3, 32, 32).astype("float32")
y = np.array([[1],[0],[1]])
output= exe.run(feed={"input": x,"label": y},
            fetch_list=[result[0]])
print(output)

#[array([0.], dtype=float32)]
