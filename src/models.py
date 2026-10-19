"""Models and explicit, batch-averaged gradients for Question 1."""
import math
import torch
from torch import nn


class ManualMLP:
    """3072 -> 100 sigmoid -> 10 logits; no autograd in training."""
    def __init__(self, device='cpu', dtype=torch.float32):
        self.w1 = torch.randn(3072, 100, device=device, dtype=dtype) / math.sqrt(3072)
        self.b1 = torch.zeros(100, device=device, dtype=dtype)
        self.w2 = torch.randn(100, 10, device=device, dtype=dtype) / math.sqrt(100)
        self.b2 = torch.zeros(10, device=device, dtype=dtype)

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x):
        h = torch.sigmoid(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2, h

    def loss_and_gradients(self, x, y):
        logits, h = self.forward(x)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -log_prob[torch.arange(len(y), device=y.device), y].mean()
        dz = log_prob.exp()
        dz[torch.arange(len(y), device=y.device), y] -= 1
        dz /= len(y)
        dw2, db2 = h.T @ dz, dz.sum(0)
        dh = (dz @ self.w2.T) * h * (1 - h)
        dw1, db1 = x.T @ dh, dh.sum(0)
        return loss, [dw1, db1, dw2, db2]

    def step(self, grads, lr):
        for p, g in zip(self.parameters(), grads):
            p.add_(g, alpha=-lr)


class LeNet5(nn.Module):
    """32x32 padded MNIST, tanh and average pooling, linear CE head."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ModernCNN(nn.Module):
    """Exactly three convolution layers, each followed by batch norm."""
    def __init__(self):
        super().__init__()
        layers = []
        for incoming, outgoing, drop in [(1, 32, .10), (32, 64, .15), (64, 128, .20)]:
            layers += [nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                       nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Dropout2d(drop)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.3),
                                        nn.Linear(128 * 4 * 4, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def freeze_backbone_training_mode(model):
    """Freeze parameters AND BN running statistics for feature extraction."""
    model.eval()
    model.fc.train()
