import torch
import torch.nn as nn
import torch.nn.functional as F

class cosine_similarity():
    def __init__(self):
        pass

    def __call__(self, x, y):
        return F.cosine_similarity(x, y, dim=1)

class nn_similarity(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(nn_similarity, self).__init__()
        self.linear_1 = nn.Sequential(
            nn.Linear(768, 2048),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
        )
        # self.linear_img = nn.Sequential(
        #     nn.Linear(768, 2048),
        #     nn.LeakyReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(2048, 1024),
        # )
        # self.linear_txt = nn.Sequential(
        #     nn.Linear(768, 2048),
        #     nn.LeakyReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(2048, 1024),
        # )
        self.linear = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, img, txt):
        # txt = self.linear_txt(txt)
        # img = self.linear_img(img)
        img = self.linear_1(img)
        txt = self.linear_1(txt)
        x = torch.abs(txt - img)
        x = self.linear(x)
        x = x.squeeze(-1)
        x = (x - 0.5) * 2
        return x


def train(model, criterion, optimizer, epochs, batch_size=64, device='cpu'):
    # 确保模型在正确的设备上
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(batch_size):
            # 生成数据并确保数据在正确的设备上
            txt = (torch.randn(batch_size, 768) if i % 2 == 0 else torch.rand(batch_size, 768)).to(device)
            img = (torch.randn(batch_size, 768) if i % 2 == 0 else torch.rand(batch_size, 768)).to(device)

            # 计算目标余弦相似度
            target = cosine_similarity()(txt, img)

            # 重置梯度
            optimizer.zero_grad()

            # 模型前向传播
            output = model(txt, img)

            # 计算损失
            loss = criterion(output, target)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            # 累积损失
            total_loss += loss.item()

        # 打印每轮的平均损失
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / batch_size}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = nn_similarity()
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam优化器

    # 训练模型
    train(model, criterion, optimizer, epochs=100, device=device)

    # 测试模型
    x = torch.rand(50, 768).to(device)
    y = torch.rand(50, 768).to(device)
    cosine = cosine_similarity()
    print(cosine(x, y))
    print(cosine(x, y).shape)

    print(model(x, y))
    print(model(x, y).shape)

    x = torch.randn(50, 768).to(device)
    y = torch.randn(50, 768).to(device)
    print(cosine(x, y))
    print(cosine(x, y).shape)

    print(model(x, y))
    print(model(x, y).shape)