import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import pdb

from dataset import CirclesDataset
from networkx import ConvAutoencoder

num_epochs = 4
batch_size = 8
device = 0

train_loader = DataLoader(CirclesDataset(), batch_size=batch_size, shuffle=True)

net = ConvAutoencoder()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    print('{}/{}'.format(epoch, num_epochs))
    tot_loss = 0
    for batch_idx, im in enumerate(train_loader):
        im = im.to(device)
        optimizer.zero_grad()

        out = net(im)
        loss = F.mse_loss(out, im)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

        # Printing
        if batch_idx % 100 == 0:
            print('{}/{}'.format(batch_idx,len(train_loader)))

    print(tot_loss/len(train_loader)/batch_size)

# Show 8 bad boys
num_show = 8
im = np.hstack([i.detach().cpu().numpy()[0] for i in im[:num_show]])
out = np.hstack([i.detach().cpu().numpy()[0] for i in out[:num_show]])
plt.imshow(np.vstack([im,out]))
plt.show()

