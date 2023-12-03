
from DataLoading import CustomDataset
from torchvision import transforms
import torch
from torch.utils.data import random_split
import pandas as pd
import random

# transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.RandomCrop(48, 4),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip()
])

# Datasets paths
train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/biasdataset'
# train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/DB/Train'
# test_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/newData/Test'

# Training dataset
total_dataset = CustomDataset(train_data_root, transform=transform)

# Manually shuffle indices
indices = list(range(len(total_dataset)))
random.shuffle(indices)

total_size = len(total_dataset)
# print(total_size)
train_size = int(0.85 * total_size)
# print(train_size)
# val_size = int(0.15 * total_size)
# print(val_size)
test_size = total_size - train_size
# print(test_size)

# ram.shuffle(total_dataset)

train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])
# print()
# Convert datasets to DataFrames
train_df = pd.DataFrame([total_dataset.data[idx] for idx in train_dataset.indices], columns=['Image_Path', 'Label'])
# val_df = pd.DataFrame([total_dataset.data[idx] for idx in val_dataset.indices], columns=['Image_Path', 'Label'])
test_df = pd.DataFrame([total_dataset.data[idx] for idx in test_dataset.indices], columns=['Image_Path', 'Label'])

# Define paths for CSV files
train_csv_path = './Facial_Expression_Detection/train_dataset.csv'
# val_csv_path = './Facial_Expression_Detection/val_dataset.csv'
test_csv_path = './Facial_Expression_Detection/test_dataset.csv'

# Save DataFrames to CSV files
train_df.to_csv(train_csv_path, index=False)
# val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)