import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


# transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.RandomCrop(48, 4),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip()
])

# Datasets paths
# train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/BiasTest_DataSet_P2'
train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/Final_dataset'

class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, general_class='', specific_class=''):
        self.root_dir = root_dir
        self.transform = transform

        self.gender_classes = os.listdir(root_dir)
        self.data = []
        gender_counter = defaultdict(int)
        age_counter = defaultdict(int)
        emotion_counter = defaultdict(int)
        for i, gender in enumerate(self.gender_classes):
            self.age_classes = os.listdir(os.path.join(root_dir, gender))
            if general_class == 'gender' and specific_class != gender:
                continue
            for j, age in enumerate(self.age_classes):
                if general_class == 'age' and specific_class != age:
                    continue
                self.emotion_classes = os.listdir(os.path.join(root_dir, gender, age))
                for k, emotion in enumerate(self.emotion_classes):
                    for file in os.listdir(os.path.join(root_dir, gender, age, emotion)):
                        self.data.append((os.path.join(root_dir, gender, age, emotion, file), k))
                        gender_counter[gender] += 1
                        age_counter[age] += 1
                        emotion_counter[emotion] += 1
        
        print("Gender Counts:", dict(gender_counter))
        print("Age Counts:", dict(age_counter))
        print("Emotion Counts:", dict(emotion_counter))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image,label
    

def store_csv(general_class, specific_class, csv_path):
    # Training dataset
    total_dataset = CustomDataset(train_data_root, transform=transform, general_class=general_class, specific_class=specific_class)
    # print(total_dataset.data)
    test_df = pd.DataFrame(total_dataset.data, columns=['Image_Path', 'Label'])
    test_df.to_csv(csv_path, index=False)


female_csv_path = './Facial_Expression_Detection/final/female_dataset.csv'
male_csv_path = './Facial_Expression_Detection/final/male_dataset.csv'
old_csv_path = './Facial_Expression_Detection/final/old_dataset.csv'
child_csv_path = './Facial_Expression_Detection/final/child_dataset.csv'
teen_csv_path = './Facial_Expression_Detection/final/teen_dataset.csv'
# female_csv_path = './Facial_Expression_Detection/main/female_dataset.csv'
# male_csv_path = './Facial_Expression_Detection/main/male_dataset.csv'
# old_csv_path = './Facial_Expression_Detection/main/old_dataset.csv'
# child_csv_path = './Facial_Expression_Detection/main/child_dataset.csv'
# teen_csv_path = './Facial_Expression_Detection/main/teen_dataset.csv'

store_csv('gender', 'Male', male_csv_path)
store_csv('gender', 'Female', female_csv_path)
store_csv('age', 'Teen', teen_csv_path)
store_csv('age', 'Child', child_csv_path)
store_csv('age', 'Old', old_csv_path)