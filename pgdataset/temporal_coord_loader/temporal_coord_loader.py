"""Load or predict temporal coordinates,
return dictionary 'coord_native', 'coord_norm'
of shape (num_frames, xy(2), num_keypoints)"""
import shutil
from pathlib import Path
from pgdataset.temporal_coord_loader.video_to_coord import VideoToCoord
from pgdataset.temporal_coord_loader.coord_persist import JointCoordPersist


class TemporalCoordLoader:

    def __init__(self, video_folder: Path, coord_folder: Path, resize_img_size: tuple):
        self.resize_img_size = resize_img_size

        self.video_to_coord = VideoToCoord(video_folder, resize_img_size)
        self.persist = JointCoordPersist(coord_folder)

    def from_video_name(self, video_name: str) -> dict:
        try:
            # keys: 'coord_native', 'coord_norm'
            coord_dict = self.persist.load(video_name)
        except FileNotFoundError:
            coord_dict = self.video_to_coord.predict(video_name)
            self.persist.save(video_name, coord_dict)
        return coord_dict

    @staticmethod
    def remove_generated_skeletons():
        p = Path("generated/coords/")
        shutil.rmtree(p)
