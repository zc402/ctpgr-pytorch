from pathlib import Path

import visdom
import numpy as np
from torch.utils.data import DataLoader
from pgdataset.s1_skeleton_coords import SkeletonCoords

ds = SkeletonCoords(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512))
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
it = iter(loader)
for n in range(1):
    try:
        next(it)
    except StopIteration:
        it = iter(loader)
        next(it)
    print(n)
