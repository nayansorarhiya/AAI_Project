
from DataLoading import CustomDataset
import pandas as pd
from torchvision import transforms
import random
from sklearn.model_selection import StratifiedKFold
from Model_Tester import start_model_test
from image_analysis import start_model_training



train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/BiasTest_DataSet_P2'
# train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/DB/Train'
# test_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/newData/Test'

# transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.RandomCrop(48, 4),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip()
])



# Training dataset
total_dataset = CustomDataset(train_data_root, transform=transform)
dataset_labels = []

# Manually shuffle indices
indices = list(range(len(total_dataset)))
random.shuffle(indices)

k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

total_accuracy = 0

for fold, (train_indices, test_indices) in enumerate(skf.split(X=total_dataset.data, y=total_dataset.labels)):
    print(f"Fold {fold + 1}")
    train_data = [total_dataset.data[i] for i in train_indices]
    test_data = [total_dataset.data[i] for i in test_indices]

    start_model_training(pd.DataFrame(train_data),None,f"trained_model_{fold+1}.pth")
    total_accuracy += start_model_test(pd.DataFrame(test_data), f"trained_model_{fold+1}.pth")

print("Average Accuracy: ", total_accuracy/k_folds)