from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CustomDatasetCSV(Dataset):
    def __init__(self, df, transform=None):
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path, label = self.data_frame.iloc[idx]
        image = Image.open(img_path).convert("L")  # Assuming grayscale images
        if self.transform:
            image = self.transform(image)
        return image, label