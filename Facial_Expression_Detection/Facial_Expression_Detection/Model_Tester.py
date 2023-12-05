import torch
import torch.cuda as cuda
from DataLoading import CustomDataset
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from DataLoadingCSV import CustomDatasetCSV
from Model import myCNNModel
from model_var1 import myCNNModel_var1
from model_var2 import myCNNModel_var2


# Pick GPU if available, else CPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def test_evaluate(model,test_loader):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend((labels.cpu().numpy()))
            all_predictions.extend((predicted.cpu()).numpy())

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_predictions)
    classification_report_str = classification_report(all_labels, all_predictions)

    # Print or log the confusion matrix and performance metrics
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report_str)
    return accuracy

# transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.RandomCrop(48, 4),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip()
])


def start_model_test(pd, save_path='Main_model.pth'):
    

    test_dataset = CustomDatasetCSV(pd, transform=transform) 

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = get_default_device()
    device

    ## Change for Different varient
    # Instantiate the model
    model = myCNNModel(1, 4)
    # model = myCNNModel_var1(1, 4)
    # model = myCNNModel_var2(1, 4)
    model = model.to(device)

    test_loader = DeviceDataLoader(DataLoader(test_dataset, 128 * 2), device)
    if cuda.is_available():
        model.load_state_dict(torch.load(save_path))
        model.eval()
        current_acc = test_evaluate(model, test_loader)
        return current_acc


def test_model_for_pd(csv_path):
    import pandas as pd
    total_data = pd.read_csv(csv_path)

    start_model_test(total_data, save_path='Final_model.pth')  # Change


if __name__ == "__main__":

    # Final model test
    test_dataset_path = "./Facial_Expression_Detection/final/test_dataset.csv"

    # TEST FOR BIAS BASED ON GENDER
    female_csv_path = './Facial_Expression_Detection/final/female_dataset.csv'
    male_csv_path = './Facial_Expression_Detection/final/male_dataset.csv'

    # TEST FOR BIAS BASED ON AGE GROUP
    old_csv_path = './Facial_Expression_Detection/final/old_dataset.csv'
    child_csv_path = './Facial_Expression_Detection/final/child_dataset.csv'
    teen_csv_path = './Facial_Expression_Detection/final/teen_dataset.csv'

    # Main model test
    # test_dataset_path = "./Facial_Expression_Detection/main/test_dataset.csv"

    # # TEST FOR BIAS BASED ON GENDER
    # female_csv_path = './Facial_Expression_Detection/main/female_dataset.csv'
    # male_csv_path = './Facial_Expression_Detection/main/male_dataset.csv'

    # # TEST FOR BIAS BASED ON AGE GROUP
    # old_csv_path = './Facial_Expression_Detection/main/old_dataset.csv'
    # child_csv_path = './Facial_Expression_Detection/main/child_dataset.csv'
    # teen_csv_path = './Facial_Expression_Detection/main/teen_dataset.csv'

    test_model_for_pd(csv_path=test_dataset_path)

    # START TESTING FOR GENDER
    print("================   TESTING FOR MALE   ================")
    test_model_for_pd(csv_path=male_csv_path)
    print("================   TESTING FOR FEMALE   ================")
    test_model_for_pd(csv_path=female_csv_path)

    # START TESTING FOR AGE
    print("================   TESTING FOR CHILD   ================")
    test_model_for_pd(csv_path=child_csv_path)
    # START TESTING FOR GENDER
    print("================   TESTING FOR TEEN   ================")
    test_model_for_pd(csv_path=teen_csv_path)
    # START TESTING FOR GENDER
    print("================   TESTING FOR Old   ================")
    test_model_for_pd(csv_path=old_csv_path)
