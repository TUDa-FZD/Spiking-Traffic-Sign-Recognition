from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """Dataset wrapper for the GTSRB traffic sign benchmark."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        csv_file: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        row = self.data_frame.iloc[idx]
        image_path = self.root_dir / row.iloc[-1]
        image = Image.open(image_path).convert("RGB")
        label = int(row.iloc[-2])

        roi = (
            int(row.iloc[2]),
            int(row.iloc[3]),
            int(row.iloc[4]),
            int(row.iloc[5]),
        )
        image = image.crop(roi)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
