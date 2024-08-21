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
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(batch_size):
            txt = (torch.randn(batch_size, 768) if i % 2 == 0 else torch.rand(batch_size, 768)).to(device)
            img = (torch.randn(batch_size, 768) if i % 2 == 0 else torch.rand(batch_size, 768)).to(device)

            target = cosine_similarity()(txt, img)

            optimizer.zero_grad()

            output = model(txt, img)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / batch_size}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn_similarity()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train(model, criterion, optimizer, epochs=100, device=device)

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
