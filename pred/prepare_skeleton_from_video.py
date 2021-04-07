from pathlib import Path

from torch.utils.data import DataLoader

from pgdataset.temporal_coord_loader.s1_skeleton import PgdSkeleton


def save():

    ds = PgdSkeleton(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for d in loader:
        pass

    ds = PgdSkeleton(Path.home() / 'PoliceGestureLong', is_train=False, resize_img_size=(512, 512))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for d in loader:
        pass