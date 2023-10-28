import os
import cv2
import pandas as pd


# Define the path to the Train folder
train_dir = 'newData/Train'

# Initialize empty lists to store image information
image_data = []

# Loop through subfolders (Angry, Boredom, Engagement, Neutral)
emotions = os.listdir(train_dir)

for emotion in emotions:
    emotion_path = os.path.join(train_dir, emotion)
    for filename in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            image_info = {
                'Image Name': filename,
                'Emotion': emotion,
                'Image Format': filename.split('.')[-1],
                'Width': image.shape[1],
                'Height': image.shape[0],
                'Size (bytes)': os.path.getsize(image_path)
            }
            image_data.append(image_info)

# Create a DataFrame from the image data
df = pd.DataFrame(image_data)

# Save the DataFrame to a CSV file
csv_file = 'image_info.csv'
df.to_csv(csv_file, index=False)

print(f"Image information saved to {csv_file}.")

