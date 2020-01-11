import numpy as np
import matplotlib.pyplot as plt
from time import time
import pdb

from torch.utils.data import Dataset
from torchvision import transforms

def append_octogonal_sym(lst,x,y):
    lst.append((x,y))
    lst.append((-x,y))
    lst.append((x,-y))
    lst.append((-x,-y))
    lst.append((y,x))
    lst.append((-y,x))
    lst.append((y,-x))
    lst.append((-y,-x))

def draw_circle(im,cx,cy,r):
    r += .5     # This makes the circle look nicer
    pts = []
    x, y = 0, int(r)
    while x <= y:
        for lx in range(0, x+1):
            append_octogonal_sym(pts, lx, y)
        x += 1
        if y*y+x*x>r*r:
            y-=1

    for i in range(x):
        for j in range(0, i+1):
            append_octogonal_sym(pts, i, j)

    for pt in pts:
        x, y = pt
        if 0 <= x+cx < im.shape[1]:
            if 0 <= y+cy < im.shape[0]:
                im[y+cy,x+cx] = 1

    return im

# Testing
'''
im = np.zeros((1,128,128))
start = time()
for i in range(10000):
    im = draw_circle(im,np.random.randint(0,128),
                        np.random.randint(0,128),
                        np.random.randint(5,10))
print(time() - start)
plt.imshow(im)
plt.show()
'''

class CirclesDataset(Dataset):
    def __init__(self, size=128, radius=5):
        self.size = size
        self.length = size**2
        self.radius = radius
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = np.unravel_index(idx, (self.size, self.size))
        im = np.zeros((self.size, self.size, 1))
        im = draw_circle(im,x,y,self.radius)
        return self.transforms(im).float()
