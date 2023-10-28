import os
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from scipy import ndimage
import random
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages


def exploreTheData():
    # started the EDA from now
    # Loading a csv to dataframe
    df = pd.read_csv("E:\\Facial_Expression_Detection\\image_info.csv")

    print(df.head())
    print()

    # Get unique image formats from the "Image Format" Column
    image_formats = df['Image Format'].unique()

    print("Unique image formats:")
    print(image_formats)

    # converting png images to jpg
    parent_directory = 'newData/Train'

# Define a function to convert PNG images to JPG in a directory
def convert_png_to_jpg(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                # Define the path for PNG and JPG files
                png_path = os.path.join(root, file)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'

                # Open and convert the PNG image to JPG
                image = Image.open(png_path)
                image = image.convert('RGB')
                image.save(jpg_path, 'JPEG')

                # Remove the original PNG file if needed
                os.remove(png_path)

def convert_all_png():
    parent_directory = 'newData/Train'
    # Iterate through the subdirectories ('Angry', 'Boredom', 'Engagement', 'Neutral')
    for emotion_folder in ['Angry', 'Boredom', 'Engagement', 'Neutral']:
        folder_path = os.path.join(parent_directory, emotion_folder)
        convert_png_to_jpg(folder_path)

    print("PNG to JPG conversion completed.")


# this method is resizing all the images to a 224 * 224
def resize_all_images(size1, size2):
    # Specify the parent directory (e.g., 'newData/Train')
    parent_directory = 'newData/Train'

    # Define the target size
    target_size = (size1, size2)

    # Define a function to resize images to the target size
    def resize_images(directory, size):
        for root, dirs, files in os.walk(directory):
            for file in files:
                image_path = os.path.join(root, file)

                # Open the image
                image = Image.open(image_path)

                # Resize the image to the target size
                image = image.resize(size, Image.BILINEAR)

                # Save the resized image, overwriting the original
                image.save(image_path)

    # Resize images in the specified subdirectories ('Angry', 'Boredom', 'Engagement', 'Neutral')
    for emotion_folder in ['Angry', 'Boredom', 'Engagement', 'Neutral']:
        folder_path = os.path.join(parent_directory, emotion_folder)
        resize_images(folder_path, target_size)

    print(f"Images resized to {target_size[0]}x{target_size[1]} pixels.")


def plotbarChart(data_type):

    parent_directory = f'newData/{data_type}'

    # List of subdirectories to process ('Angry', 'Boredom', 'Engagement', 'Neutral')
    subdirectories = ['Angry', 'Boredom', 'Engagement', 'Neutral']

    # Initialize a dictionary to store the count of images in each category
    category_counts = {}
    # Count the number of images in each category
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(parent_directory, subdirectory)
        image_files = [filename for filename in os.listdir(subdirectory_path) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        category_counts[subdirectory] = len(image_files)

    # Create a bar chart
    plt.bar(category_counts.keys(), category_counts.values())
    plt.xlabel('Categories')
    plt.ylabel('Number of Photos')
    plt.title('Number of Photos in Each Category')
    plt.xticks(rotation=45)  # Rotate category labels for readability

    if data_type == "Train":
        plt.ylim(400,420)
        plt.yticks(range(400,420,5))
    else:
        plt.ylim(140,160)
        plt.yticks(range(140,160,5))

    # Show the bar chart
    plt.show()

def intensity_for_random():
    # Directory containing the subdirectories of images (e.g., 'newData/Train')
    parent_directory = 'newData/Train'

    # List of subdirectories to process ('engagement', 'angry', 'neutral', 'boredom')
    subdirectories = ['engagement', 'angry', 'neutral', 'boredom']

    # Emotion labels based on subdirectories
    emotion_labels = {
        'engagement': 'Engagement',
        'angry': 'Angry',
        'neutral': 'Neutral',
        'boredom': 'Boredom'
    }

    # Create a 5x5 grid for displaying images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    # Randomly select 25 images and their corresponding emotions
    random_image_emotions = []

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(parent_directory, subdirectory)

        # List all image files in the subdirectory
        image_files = [os.path.join(subdirectory_path, filename) for filename in os.listdir(subdirectory_path) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Check if there are any images in the subdirectory
        if image_files:
            # Randomly select an image from each subdirectory and its emotion label
            random_image = random.choice(image_files)
            emotion_label = emotion_labels[subdirectory]
            random_image_emotions.append((random_image, emotion_label))

    # If you have fewer than 25 images, repeat randomly selecting from the available images
    while len(random_image_emotions) < 25:
        random_subdirectory = random.choice(subdirectories)
        subdirectory_path = os.path.join(parent_directory, random_subdirectory)
        image_files = [os.path.join(subdirectory_path, filename) for filename in os.listdir(subdirectory_path) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Check if there are any images in the subdirectory
        if image_files:
            # Randomly select an image from each subdirectory and its emotion label
            random_image = random.choice(image_files)
            emotion_label = emotion_labels[random_subdirectory]
            random_image_emotions.append((random_image, emotion_label))

    # Shuffle the selected images and emotions to ensure randomness
    random.shuffle(random_image_emotions)
    fig_hist = plt.figure(figsize=(10, 5))
    for i, (image_path, emotion_label) in enumerate(random_image_emotions):
        image = plt.imread(image_path)
        row, col = divmod(i, 5)
        axes[row, col].imshow(image)
        axes[row, col].set_title(emotion_label, fontsize=10)
        axes[row, col].axis('off')

        hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))

        # Normalize the histogram for grayscale images
        hist = hist / hist.sum()

        plt.figure(fig_hist.number)
        plt.subplot(5, 5, i + 1)
        plt.plot(hist, color='black')

        # For RGB images, calculate and plot histograms for each channel
        if len(image.shape) == 3 and image.shape[2] == 3:
            r_hist, _ = np.histogram(image[:, :, 0].ravel(), bins=256, range=(0, 256))
            g_hist, _ = np.histogram(image[:, :, 1].ravel(), bins=256, range=(0, 256))
            b_hist, _ = np.histogram(image[:, :, 2].ravel(), bins=256, range=(0, 256))
            r_hist = r_hist / r_hist.sum()
            g_hist = g_hist / g_hist.sum()
            b_hist = b_hist / b_hist.sum()

            plt.plot(r_hist, color='red', alpha=0.7)
            plt.plot(g_hist, color='green', alpha=0.7)
            plt.plot(b_hist, color='blue', alpha=0.7)

    plt.tight_layout()
    plt.show()

def barchartforImagesize():
    df = pd.read_csv("E:\\Facial_Expression_Detection\\image_info.csv")
    # Extract the "Size (bytes)" column for plotting
    image_sizes = df['Size (bytes)']
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, image_sizes, color='b', alpha=0.7, label='Image Sizes')
    plt.xlabel('Image Index')
    plt.ylabel('Size (bytes)')
    plt.title('Image Sizes')
    plt.legend()
    plt.show()

def plotRandom():
    # Directory containing the subdirectories of images (e.g., 'newData/Train')
    parent_directory = 'newData/Train'

    # List of subdirectories to process ('engagement', 'angry', 'neutral', 'boredom')
    subdirectories = ['engagement', 'angry', 'neutral', 'boredom']

    # Emotion labels based on subdirectories
    emotion_labels = {
        'engagement': 'Engagement',
        'angry': 'Angry',
        'neutral': 'Neutral',
        'boredom': 'Boredom'
    }

    # Create a 5x5 grid for displaying images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    # Randomly select 25 images and their corresponding emotions
    random_image_emotions = []

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(parent_directory, subdirectory)

        # List all image files in the subdirectory
        image_files = [os.path.join(subdirectory_path, filename) for filename in os.listdir(subdirectory_path) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Check if there are any images in the subdirectory
        if image_files:
            # Randomly select an image from each subdirectory and its emotion label
            random_image = random.choice(image_files)
            emotion_label = emotion_labels[subdirectory]
            random_image_emotions.append((random_image, emotion_label))

    # If you have fewer than 25 images, repeat randomly selecting from the available images
    while len(random_image_emotions) < 25:
        random_subdirectory = random.choice(subdirectories)
        subdirectory_path = os.path.join(parent_directory, random_subdirectory)
        image_files = [os.path.join(subdirectory_path, filename) for filename in os.listdir(subdirectory_path) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Check if there are any images in the subdirectory
        if image_files:
            # Randomly select an image from each subdirectory and its emotion label
            random_image = random.choice(image_files)
            emotion_label = emotion_labels[random_subdirectory]
            random_image_emotions.append((random_image, emotion_label))

    # Shuffle the selected images and emotions to ensure randomness
    random.shuffle(random_image_emotions)

    for i, (image_path, emotion_label) in enumerate(random_image_emotions):
        image = plt.imread(image_path)
        row, col = divmod(i, 5)
        axes[row, col].imshow(image)
        axes[row, col].set_title(emotion_label, fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


# this method is just used to explore the data so we can clean it properly
exploreTheData()

# this method is used to convert all the png files to jpg
convert_all_png()

# this method is used to resize the images. Here we have passed 224 * 224.
resize_all_images(224, 224)

# this is the bar chart which shows the number of images for all the 4 emotions in both train and test dataset
plotbarChart("Test")
plotbarChart("Train")

# this method is used to plot 25 random images in 5 * 5 grid
plotRandom()

# this method is used to show the RGB intensity for 25 random images
intensity_for_random()

# this method is used to show the different sizes of images
barchartforImagesize()




