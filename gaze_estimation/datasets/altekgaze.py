import pathlib
from typing import Callable, Tuple
import numpy as np
import csv
import torch
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm

class AltekDataset(Dataset):
    def __init__(self, dataset_csv_path: pathlib.Path, transform: Callable):
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        
        images = []
        poses = []
        gazes = []
        
        with open(dataset_csv_path, newline='') as csvfile:
            for row in tqdm(csv.DictReader(csvfile)):
                images.append(cv2.imread(row['Wrapped_path'], 0))
                gazes.append((float(row['Gaze_pitch']), float(row['Gaze_yaw'])))
                poses.append((float(row['Head_pitch']), float(row['Head_yaw'])))
                    
        self.images = np.array(images)
        self.poses = np.array(poses, dtype='float32')
        self.gazes = np.array(gazes, dtype='float32')

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        return image, pose, gaze

    def __len__(self) -> int:
        return len(self.images)

if __name__ == '__main__':
    import torchvision
    transform = torchvision.transforms.Compose([torch.from_numpy])
    AltekDataset('training_data.csv', transform)