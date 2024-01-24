# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
#
# # 定义数据集的转换（预处理）
# transform = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),  # 调整图像大小
#         transforms.ToTensor(),  # 转换为张量
#     ]
# )
#
# # 下载CelebA数据集（如果尚未下载），并应用转换
# celeba_dataset = datasets.CelebA(
#     root="./data", split="all", transform=transform, download=True
# )
#
# # 创建数据加载器
# batch_size = 1
# data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)
#
# # 查看数据加载器的一个批次
# for batch in data_loader:
#     images, labels = batch
#     # 在这里添加你的训练代码，images是输入的人脸图像，labels是一些属性标签（可选）
#     plt.imshow(images)
