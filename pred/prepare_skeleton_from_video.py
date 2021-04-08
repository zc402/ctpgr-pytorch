from pathlib import Path

from torch.utils.data import DataLoader

from pgdataset.s1_temporal_coord_dataset import TemporalCoordDataset


def save():

    ds = TemporalCoordDataset(Path.home() / 'PoliceGestureLong', is_train=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for d in loader:
        pass

    ds = TemporalCoordDataset(Path.home() / 'PoliceGestureLong', is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for d in loader:
        pass