"""Save or load joint coordinates, shape (num_frames, xy(2), num_keypoints)"""
import pickle

class JointCoordPersist:
    def __init__(self, coord_folder):
        self.coord_folder = coord_folder
        coord_folder.mkdir(parents=True, exist_ok=True)
        pass

    def load(self, file_name: str) -> dict:
        pkl_path = self.coord_folder / file_name
        pkl_path = pkl_path.with_suffix('.pkl')
        if not pkl_path.exists():
            raise FileNotFoundError()
        with pkl_path.open('rb') as pickle_file:
            coords = pickle.load(pickle_file)
        return coords

    def save(self, file_name: str, coord: dict):
        pkl_path = self.coord_folder / file_name
        pkl_path = pkl_path.with_suffix('.pkl')
        with pkl_path.open('wb') as pickle_file:
            pickle.dump(coord, pickle_file)
