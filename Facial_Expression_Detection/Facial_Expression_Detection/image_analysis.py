import os
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from DataLoading import CustomDataset
from DataLoadingCSV import CustomDatasetCSV
from Classification import ImageClassificationBase
from Model import myCNNModel
from model_var1 import myCNNModel_var1
from model_var2 import myCNNModel_var2
import pandas as pd

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
    
def evaluate(model, val_loader, accuracies_list):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            accuracies_list.append(accuracy)
    print('Test Accuracy of the model on the test images: {} %'.format(accuracy))

    outputs1 = [model.validation_step(batch) for batch in val_loader]
    # return true_labels_list,predicted_labels_list
    
    return model.validation_epoch_end(outputs1),accuracy

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

    plot_accuracies(history)
    plot_losses(history)

    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD,
                  early_stopping_patience=None, save_path='trained_model.pth'):
    torch.cuda.empty_cache()
    history = []
    accuracies_list = []
    losses_list = []    

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    # Initialize early stopping variables
    best_val_loss = float('inf')
    early_stopping_counter_a = 0
    early_stopping_counter_l = 0
    best_accuracy = 0.0000
    current_accuracy = 0.0000
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

            #Train Accuracy
            total = batch[1].size(0)
            outputs = model(batch[0])
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch[1]).sum().item()
            accuracy = (correct / total) * 100
            accuracies_list.append(accuracy)
            losses_list.append(loss.item())

        # Validation phase
        with torch.no_grad():
            result,accuracy = evaluate(model, val_loader, accuracies_list)
            current_accuracy = accuracy
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
        # print(current_accuracy)
        # print(best_accuracy)
        # print(result['val_loss'])
        # print(best_val_loss)

        # Early stopping check
        
        val_loss = result['val_loss']
        
        if best_accuracy < current_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), save_path)
            # torch.save(model.state_dict(), 'trained'+str(epoch)+'.pth')
            early_stopping_counter_a = 0
        else:
            early_stopping_counter_a += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter_l = 0
        else:
            early_stopping_counter_l += 1

        if early_stopping_counter_a >= early_stopping_patience and early_stopping_counter_l >= early_stopping_patience:
            print(f'Early stopping after epoch {epoch + 1}')
            break

    return history

# plot_accuracies(history)
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. Number of Epochs')
    plt.show()


# plot_losses(history)
def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. Number of Epochs')
    plt.show()
    
def start_model_training(total_data, save_path = 'trained_model.pth'):
    # transformations for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.RandomCrop(48, 4),
        transforms.Normalize((0.5), (0.5)),
        transforms.RandomHorizontalFlip()
    ])
    total_list = list(total_data.itertuples(index=False, name=None))


    train_size = int(len(total_list)*.85)
    val_size = len(total_list) - train_size

    train_set, val_set = random_split(total_list, [train_size, val_size])

    train_dataset = CustomDatasetCSV(pd.DataFrame([total_list[idx] for idx in train_set.indices], columns=['Image_Path', 'Label']), transform=transform) 
    val_dataset = CustomDatasetCSV(pd.DataFrame([total_list[idx] for idx in val_set.indices], columns=['Image_Path', 'Label']), transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = get_default_device()
    device

    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)


    # Running for a Base Model


    print("Base Model Running")
    print(" ")

    # Instantiate the model

    model = myCNNModel(1, 4)
    # model = myCNNModel_var1(1, 4)    ##Extra layer
    # model = myCNNModel_var2(1, 4)      ## kernal size 5
    model = model.to(device)

    # torch.save(model.state_dict(), 'trained_model.pth')
    # history = [evaluate(model, val_dl)]
    # history

    epochs = 15
    max_lr = 0.0001
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    early_stopping_patience = 10

    # %%time
    history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                            grad_clip=grad_clip,
                            weight_decay=weight_decay,
                            opt_func=opt_func,
                            early_stopping_patience=early_stopping_patience,
                            save_path=save_path)
    return history



if __name__ == "__main__":

    # Datasets paths
    train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/biasdataset'
    # train_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/DB/Train'
    # test_data_root = 'S:/concordia/all_terms/fall_2023/AAI/Phase3/AAI_Project/Facial_Expression_Detection/Facial_Expression_Detection/newData/Test'




    # # Training dataset
    # total_dataset = CustomDataset(train_data_root, transform=transform)

    # total_size = len(total_dataset)
    # # print(total_size)
    # train_size = int(0.7 * total_size)
    # # print(train_size)
    # val_size = int(0.15 * total_size)
    # # print(val_size)
    # test_size = total_size - train_size - val_size
    # # print(test_size)

    train_dataset_path = "./Facial_Expression_Detection/train_dataset.csv"
    import pandas as pd
    total_data = pd.read_csv(train_dataset_path)
    history = start_model_training(total_data)
    plot_accuracies(history)
    plot_losses(history)


# Testing Saved Model

# test_loader = DeviceDataLoader(DataLoader(test_dataset, 128 * 2), device)
# if cuda.is_available():
#     load_model = model.load_state_dict(torch.load(save_path))
#     model.eval()
#     result = test_evaluate(model, test_loader)
#     result



# # Running for a Model Variation 1 : (Kernel Size : 5)


# print("Variation 1 Model Running")
# print(" ")


# # Instantiate the model
# model = myCNNModel_var1(1, 4)

# accuracies_list = []
# losses_list = []

# torch.save(model.state_dict(), 'trained_model.pth')
# history = [evaluate(model, val_dl)]
# # history

# epochs = 12
# max_lr = 0.0001
# grad_clip = 0.1
# weight_decay = 1e-4
# opt_func = torch.optim.Adam

# early_stopping_patience = 10

# # %%time
# history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
#                          grad_clip=grad_clip,
#                          weight_decay=weight_decay,
#                          opt_func=opt_func,
#                          early_stopping_patience=early_stopping_patience)

# test_loader = DeviceDataLoader(DataLoader(test_dataset, 128 * 2), device)
# result = evaluate(model, test_loader)
# result

# all_labels = []
# all_predictions = []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         all_labels.extend(labels.numpy())
#         all_predictions.extend(predicted.numpy())

# # Generate confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_predictions)

# # Calculate accuracy, precision, recall, and F1-score
# accuracy = accuracy_score(all_labels, all_predictions)
# classification_report_str = classification_report(all_labels, all_predictions)

# # Print or log the confusion matrix and performance metrics
# print("Confusion Matrix:")
# print(conf_matrix)
# print("\nAccuracy:", accuracy)
# print("\nClassification Report:")
# print(classification_report_str)

# plot_accuracies(history)
# plot_losses(history)



# # Running for a Model Variation 2 : (increase a layer)


# print("Variation 2 Model Running")
# print(" ")


# # Instantiate the model
# model = myCNNModel_var2(1, 4, 3)

# accuracies_list = []
# losses_list = []

# torch.save(model.state_dict(), 'trained_model.pth')
# history = [evaluate(model, val_dl)]
# # history

# epochs = 12
# max_lr = 0.0001
# grad_clip = 0.1
# weight_decay = 1e-4
# opt_func = torch.optim.Adam

# early_stopping_patience = 10

# # %%time
# history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
#                          grad_clip=grad_clip,
#                          weight_decay=weight_decay,
#                          opt_func=opt_func,
#                          early_stopping_patience=early_stopping_patience)

# test_loader = DeviceDataLoader(DataLoader(test_dataset, 128 * 2), device)
# result = evaluate(model, test_loader)
# result

# all_labels = []
# all_predictions = []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)+

#         all_labels.extend(labels.numpy())
#         all_predictions.extend(predicted.numpy())

# # Generate confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_predictions)

# # Calculate accuracy, precision, recall, and F1-score
# accuracy = accuracy_score(all_labels, all_predictions)
# classification_report_str = classification_report(all_labels, all_predictions)

# # Print or log the confusion matrix and performance metrics
# print("Confusion Matrix:")
# print(conf_matrix)
# print("\nAccuracy:", accuracy)
# print("\nClassification Report:")
# print(classification_report_str)

# plot_accuracies(history)
# plot_losses(history)