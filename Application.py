import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import torchvision.models as models

# Define the path to your saved model (.pth file)
model_path = 'path/to/your/saved_model.pth'

# Load the saved model
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define image transformation for input
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_emotion(model, image_path, transform):
    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        emotion_pred = torch.argmax(probabilities).item()

    return emotion_pred

# Example: Predict emotion for an individual image
image_path = 'path/to/your/image.jpg'
predicted_emotion = predict_emotion(model, image_path, transform)

# Example: Predict emotions for a dataset
def predict_emotions_for_dataset(model, dataset_path, transform):
    # Assuming the dataset_path contains images for prediction
    # You can iterate through the images and use the predict_emotion function
    # to get predictions for each image.
    # Example:
    for image_filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_filename)
        predicted_emotion = predict_emotion(model, image_path, transform)
        print(f"Image: {image_filename}, Predicted Emotion: {predicted_emotion}")

# Example usage for predicting emotions for a dataset
dataset_path = 'path/to/your/dataset'
predict_emotions_for_dataset(model, dataset_path, transform)
