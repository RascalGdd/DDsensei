import torch
from torch import nn, optim, autograd
import numpy as np
import random

h_dim = 400
batch_size = 128

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = self.net(z)
        return output

def data_generator():
    scale = 2.
    centers = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1./np.sqrt(2),1./np.sqrt(2)),
        (-1./np.sqrt(2),1./np.sqrt(2)),
        (1./np.sqrt(2),-1./np.sqrt(2)),
        (-1./np.sqrt(2),-1./np.sqrt(2))
    ]

    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.asarray(dataset).astype(np.float32)
        dataset = dataset/1.414
        yield dataset

def main():
    torch.manual_seed(23)
    np.random.seed(23)
    random.seed(23)

    data_iter = data_generator()
    x = next(data_iter)
    G = Generator().cuda()
    D = Discriminator().cuda()
    optim_G = optim.Adam(G.parameters(),lr=5e-4,betas=(0.5,0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    for epoch in range(5000):
        for _ in range(5):
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            predictr = D(xr)
            lossr = -predictr.mean()
            z = torch.randn(batch_size, 2).cuda()
            xf = G(z).detach()
# detach的用处是终止梯度，防止梯度传递到G，因为这里我们只优化D
            predictf = D(xf)
            lossf = predictf.mean()
            loss_D = lossr + lossf
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        z = torch.randn(batch_size, 2).cuda()
        xf = G(z)
        predictf = D(xf)
# 为什么这里不用detach？ 因为梯度传递方向是 z→G→D，如果从D停止梯度，那就传递不到G了，就没法优化了
        loss_G = -predictf.mean()
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            print(loss_D.item(),loss_G.item())




if __name__ == '__main__':
    main()



