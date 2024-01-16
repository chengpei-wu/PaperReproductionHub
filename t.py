import torch

# 定义权重矩阵
weight = torch.tensor(3, requires_grad=True, dtype=float)

# 创建 x，它是 weight 的引用
# x = weight

# w=3, y = 2*w, z = y^2, z = (2w)^2 = 4w^2, z' = 8w = 24

# 对 x 进行操作
y = 2 * weight

z = y**2

# 反向传播
z.backward()
# 输出梯度
print(weight.grad)  # 输出梯度值
