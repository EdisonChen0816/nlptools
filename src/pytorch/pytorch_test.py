# encoding=utf-8
import torch
import numpy as np
import torch.nn as nn

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.zeros(5, 3).long()
print(x.dtype)

x = torch.tensor([5.5, 3])
print(x)


x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.shape, x.size())

y = torch.rand(5, 3)
print(x + y, torch.add(x, y))


result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)  # in-place, 结果保存在y里面，在y里面操作
print(y)

print(x[1:, 1:])

# 如果你希望resize/reshape一个tensor，可以使用torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(2, 8)
print(z)

# 如果你有一个只有一个元素的tensor，使用.item()方法可以把里面的value变成python数值
x = torch.randn(1)
print(x)
print(dir(x))
print(x.item())

print(z.transpose(1, 0))




## Numpy和Tensor之间的转化

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)  # a和b共享内存

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a, b)

a = a + 1
print(a, b)

# CUDA Tensors
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))

y.to('cpu').data.numpy()
y.cpu().data.numpy()

N, D_in, H, D_out = 64, 1000, 100, 10
model = nn.Sequential(
    nn.Linear(D_in, H, bias=False),
    nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=False)
)

print(model)
print(model[0])
print(model[0].weight)


class TowLayerNet(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(TowLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H, bias=False)
        self.linear2 = nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TowLayerNet(D_in, H, D_out)
model(x)