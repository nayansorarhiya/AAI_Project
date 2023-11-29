import torch
from torchvision import transforms
from Model import myCNNModel, Category
from PIL import Image
import torch.nn.functional as F
import os
import torchvision.models as models

# Define the path to your saved model (.pth file)
model_path = 'C:/Users/admin/Downloads/AK_08_P2/trained_model.pth'

emotion_dict = {
    0: "Angry",
    1: "Boredom",
    2: "Engagement",
    3: "Neutral"
}

# Load the saved model
loaded_model = myCNNModel(1, 4)  # Assuming ResNet9 is your model class
loaded_model.load_state_dict(torch.load('trained_model.pth'))
loaded_model.eval()

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

# C:\\Users\\admin\\Downloads\\AK_08_P2\\Facial_Expression_Detection\\Facial_Expression_Detection\\newData\\Train\\Engagement\\engagement0.jpg

# C:\\Users\\admin\\Downloads\\AK_08_P2\\Facial_Expression_Detection\\Facial_Expression_Detection\\newData\\Test\\temp\\ffhq_105.jpg

# Example: Predict emotion for an individual image
image_path = 'C:\\Users\\admin\\Downloads\\AK_08_P2\\Facial_Expression_Detection\\Facial_Expression_Detection\\DB\\Train\\Angry\\image0000006.jpg'
predicted_emotion = predict_emotion(loaded_model, image_path, transform)
index = Category(image_path,predicted_emotion)

print(f"Predicted Emotion: {emotion_dict[index]}")
