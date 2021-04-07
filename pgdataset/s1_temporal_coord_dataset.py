from torch.utils.data import Dataset
from pathlib import Path

from constants.enum_keys import PG
from .s0_label_loader import LabelLoader
from .temporal_coord_loader import TemporalCoordLoader

class TemporalCoordDataset(Dataset):
    def __init__(self, data_path: Path, is_train: bool):
        self.resize_img_size = (512, 512)
        if is_train:
            self.coord_folder = Path("generated/coords/train/")
            self.video_folder = data_path / "train"
        else:
            self.coord_folder = Path("generated/coords/test/")
            self.video_folder = data_path / "test"

        self.label_loader = LabelLoader(data_path, is_train)
        self.coord_loader = TemporalCoordLoader(self.video_folder, self.coord_folder, self.resize_img_size)

    def __len__(self):
        return self.label_loader.num_videos()

    def __getitem__(self, index) -> dict:
        # {PG.VIDEO_NAME, PG.VIDEO_PATH, PG.GESTURE_LABEL, PG.NUM_FRAMES}
        label_dict = self.label_loader[index]
        # {'coord_native', 'coord_norm'}
        coord_dict = self.coord_loader.from_video_name(label_dict[PG.VIDEO_NAME])
        res_dict = {}
        res_dict.update(label_dict)
        res_dict.update(coord_dict)
        return res_dict

