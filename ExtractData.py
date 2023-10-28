# Importing all necessary libraries
import pandas as pd
import cv2
import os

# Load the CSV file into a DataFrame
df = pd.read_csv('E:\\Facial_Expression_Detection\\Data\\Labels\\TrainLabels.csv')

boredom = df['ClipID'][(df['Boredom'] > 1) & (df['Engagement'] < 2)]
engagement = df['ClipID'][(df['Boredom'] == 0) & (df['Engagement'] > 2)]


array_boredom = boredom.values[:135]
array_engagement = engagement.values[-135:]


try:
    # Creating folders for "Train" directory
    if not os.path.exists('newData/Train/Engagement'):
        os.makedirs('newData/Train/Engagement')

    if not os.path.exists('newData/Train/Boredom'):
        os.makedirs('newData/Train/Boredom')

    # Creating folders for "Test" directory
    if not os.path.exists('newData/Test/Engagement'):
        os.makedirs('newData/Test/Engagement')

    if not os.path.exists('newData/Test/Boredom'):
        os.makedirs('newData/Test/Boredom')

except OSError:
    print('Error: Creating directory of data')

# Taking Frames out of the video from a Training dataset
PATH = "E:\\Facial_Expression_Detection\\Data\\DataSet"
Type = "Train"

# working on engagement dataset
engagement_image_count = 0
for index in range(len(array_engagement)):
    element = array_engagement[index]

    cam = cv2.VideoCapture(os.path.join(PATH, Type, element[:6], element[:-4], element))

    # frame
    frame_rate = 1 / 3  # one frame every 3 seconds
    currentframe = 0

    while(True):

        # reading from frame
        ret, frame = cam.read()
        time_in_seconds = currentframe * frame_rate

        if ret and int(time_in_seconds) % 3 == 0:

            # if video is still left continue creating images
            name = f'newData/Train/Engagement/engagement{engagement_image_count}.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that i will show how many frames are created
            currentframe += 1
            engagement_image_count += 1

        else:
            break


    # working on boredom dataset
    boredom_image_count = 0
    for index in range(len(array_boredom)):
        element = array_boredom[index]

        cam = cv2.VideoCapture(os.path.join(PATH, Type, element[:6], element[:-4], element))

        # frame
        frame_rate = 1 / 3  # one frame every 3 seconds
        currentframe = 0

        while (True):

            # reading from frame
            ret, frame = cam.read()
            time_in_seconds = currentframe * frame_rate

            if ret and int(time_in_seconds) % 3 == 0:

                # if video is still left continue creating images
                name = f'newData/Train/Boredom/boredom{boredom_image_count}.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that i will show how many frames are created
                currentframe += 1
                boredom_image_count += 1

            else:
                break

        # Release all space and windows once done
        cam.release()
cv2.destroyAllWindows()


# working on test dataset

# Load the CSV file into a DataFrame
df = pd.read_csv('E:\\Facial_Expression_Detection\\Data\\Labels\\TestLabels.csv')

boredom = df['ClipID'][(df['Boredom'] > 1) & (df['Engagement'] < 2)]
engagement = df['ClipID'][(df['Boredom'] == 0) & (df['Engagement'] > 2)]

array_boredom = boredom.values[:50]
array_engagement = engagement.values[-50:]


Type = "Test"

# working on engagement dataset
engagement_image_count = 0
for index in range(len(array_engagement)):
    element = array_engagement[index]

    cam = cv2.VideoCapture(os.path.join(PATH, Type, element[:6], element[:-4], element))

    # frame
    frame_rate = 1 / 3  # one frame every 3 seconds
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()
        time_in_seconds = currentframe * frame_rate

        if ret and int(time_in_seconds) % 3 == 0:

            # if video is still left continue creating images
            name = f'newData/Test/Engagement/engagement{engagement_image_count}.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that i will show how many frames are created
            currentframe += 1
            engagement_image_count += 1

        else:
            break

    # working on boredom dataset
    boredom_image_count = 0
    for index in range(len(array_boredom)):
        element = array_boredom[index]

        cam = cv2.VideoCapture(os.path.join(PATH, Type, element[:6], element[:-4], element))

        # frame
        frame_rate = 1 / 3  # one frame every 3 seconds
        currentframe = 0

        while (True):

            # reading from frame
            ret, frame = cam.read()
            time_in_seconds = currentframe * frame_rate

            if ret and int(time_in_seconds) % 3 == 0:

                # if video is still left continue creating images
                name = f'newData/Test/Boredom/boredom{boredom_image_count}.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that i will show how many frames are created
                currentframe += 1
                boredom_image_count += 1

            else:
                break

        # Release all space and windows once done
        cam.release()
cv2.destroyAllWindows()












