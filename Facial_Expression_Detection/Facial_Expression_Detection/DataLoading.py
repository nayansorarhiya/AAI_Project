import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.gender_classes = os.listdir(root_dir)
        self.data = []
        for i, gender in enumerate(self.gender_classes):
            self.age_classes = os.listdir(os.path.join(root_dir, gender))
            for j, age in enumerate(self.age_classes):
                self.emotion_classes = os.listdir(os.path.join(root_dir, gender, age))
                for k, emotion in enumerate(self.emotion_classes):
                    for file in os.listdir(os.path.join(root_dir, gender, age, emotion)):
                        self.data.append((os.path.join(root_dir, gender, age, emotion, file), k))
        self.labels = [label for _, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image,label
    
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = sorted(os.listdir(root_dir))
#         self.data = []
#         for i, class_name in enumerate(self.classes):
#             class_path = os.path.join(root_dir, class_name)
#             for file in os.listdir(class_path):
#                 self.data.append((os.path.join(class_path, file), i))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path, label = self.data[idx]
#         image = Image.open(img_path).convert("L")
#         if self.transform:
#             image = self.transform(image)
#         return image, label
    

