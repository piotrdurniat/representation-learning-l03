from torch import nn, Tensor
from torch.nn import functional as F


class SmallConvnet(nn.Module):
    """Small ConvNet (avoids heavy computation)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, plain_last: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if not plain_last:
            self.net.append(nn.BatchNorm1d(output_dim))
            self.net.append(nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
