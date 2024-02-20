# -*- coding: utf-8 -*-
# coding=utf-8
from mindspore import Tensor, Parameter
import numpy as np
import pickle
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, XavierUniform

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import Parameter, Tensor
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class CustomOptimizer(nn.Cell):
    def __init__(self, params, learning_rate, max_grad_norm, lr_decay=1, start_decay_steps=None, decay_steps=None,
                 adagrad_accum=0.0, decay_method=None, warmup_steps=4000, model_size=None):
        super(CustomOptimizer, self).__init__(auto_prefix=False)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.assignadd = P.AssignAdd()
        self.assign = P.Assign()
        self.max_grad_norm = Tensor(max_grad_norm, mstype.float32)
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        # self.adagrad_accum = Tensor(adagrad_accum, mstype.float32)
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        if isinstance(params, list) and all(isinstance(p, Parameter) for p in params):
            self.params = params
            self.learning_rate = Tensor(learning_rate, mstype.float32)
            self.grad_accum = Parameter(Tensor(0.0, mstype.float32), name="grad_accum")
            self.adagrad_accum = Parameter(Tensor(adagrad_accum, mstype.float32), name="adagrad_accum")
        else:
            raise ValueError("params type error")

    def construct(self, grad):
        norm_grad = grad / (self.sqrt(self.square(grad)) + 1e-6)
        norm_grad = P.clip_by_norm(norm_grad, self.max_grad_norm)

        # Update adagrad accumulators.
        new_adagrad_accum = self.assignadd(self.adagrad_accum, self.square(norm_grad))

        # Compute the learning rate.
        learning_rate = self.learning_rate / (self.sqrt(new_adagrad_accum) + 1e-6)

        # Adjust the learning rate based on the provided parameters.
        if self.decay_method == "noam":
            step_num = P.AssignAdd()(self.grad_accum, 1.0)
            warmup_arg1 = step_num ** (-0.5)
            warmup_arg2 = step_num * (self.warmup_steps ** (-1.5))
            learning_rate *= (self.model_size ** (-0.5)) * P.minimum()(warmup_arg1, warmup_arg2)

        # Apply the update to the parameters.
        new_param = P.AssignSub()(self.params, learning_rate * norm_grad)

        return new_param


def print_elements(var):
    if isinstance(var, (list, tuple)):
        for elem in var:
            print(elem)
    elif isinstance(var, dict):
        for key, value in var.items():
            print(f'{key}: {value}')
    elif hasattr(var, '__dict__'):
        for attr, value in vars(var).items():
            # print(attr, type(value))
            print(f'{attr}: {value}')
    else:
        print(var)


# with open("./test.pkl", 'rb') as f:
#     ms_datas = pickle.load(f)
#     newTensor = Tensor(ms_datas)
#     print(newTensor.shape, newTensor.dtype)

from mindspore import Tensor, ops
import numpy as np

x1 = np.array([[1.1998, 0.8533, -0.0655, 0.6623, -0.6258, 0.4602, -0.8213, 0.5070,
                -0.4033, 0.7897],
               [-0.1918, 0.5189, 0.1178, -0.6766, 0.1553, -0.3086, -0.0322, -0.1294,
                0.2166, 0.1706]]).astype(np.float32)
x2 = np.array([0.6463, 0.0423]).astype(np.float32)
input_x = (Tensor(x1), Tensor(x2))
out = ops.clip_by_global_norm(input_x, 1.0)
print("1: ", out)
print("2: ", ops.clip_by_global_norm(input_x, 2.0))
print("3: ", ops.clip_by_global_norm(input_x, 3.0))
print("4: ", ops.clip_by_global_norm(input_x, 4.0))
print("5: ", ops.clip_by_global_norm(input_x, 5.0))

import torch
import torch.nn as nn

torch.manual_seed(42)
# 创建一个简单的线性模型
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 创建一些输入数据
input = torch.randn(3, 10)
target = torch.randn(3, 2)

# 前向传播
output = model(input)

# 计算损失
loss = nn.MSELoss()(output, target)

# 反向传播，计算梯度
loss.backward()

# 打印裁剪前的梯度
print("Gradients before clipping")
for param in model.parameters():
    print(param.grad)

# 裁剪梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

# 打印裁剪后的梯度
print("Gradients after clipping")
for param in model.parameters():
    print(param.grad)

# 更新权重
optimizer.step()

# 清除所有优化器的grad以防止累积
optimizer.zero_grad()
