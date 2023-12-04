----------------------------------
|---- A.I. ducation Analytics ----|
----------------------------------



[1] Imports are given below that are imported:
    - pip3 install pandas
    - pip3 install opencv-python
    - pip3 install pillow
    - pip3 install matplotlib
    - pip3 install torch
    - pip3 install torchvision
    - pip install scikit-learn
    - pip3 install PIL
    - pip3 install matplotlib

[2] project_directory Structure

├── BiasTest_dataset_P2/                     // class_data contains the data set(images) for emotions
│   ├── Male/                     IMAGES 1513 
│   │     Child,Old,Teen
|   |      ├── Anger
|   │      ├── Boredom
|   │      ├── Engagement
|   │      └── Neutral
│   └── Female/                   IMAGES 757
|          Child,Old,Teen                         
|           ├── Anger
|           ├── Boredom
|           ├── Engagement
|           └── Neutral
│
|
├── LabelledData.py            // Takes each class data from train and labels it with meta data and stores the data into image_info.csbv
│
├── EDA.py        // Contains code for all the plots using Matplotlib
│
├── EDA.py             // for resizing the images and converting png to jpeg
│
├── ExtractData.py                   // fetches data from the easy data set which is stored in local
│
├── image_info.csv                  // file for meta data
│
├── image_analysis.py               // training the model and testing the dataset
│
├── classify_image.py               // test image category by trained model.
│
│
├─── image_analysis_var1             // Changed number of con layers.
│
├── image_analysis_var2             // Changed kernel size of based version.
│
|
├── DataLoading.py                  // get image path and label from dataset repository structure 
|
├── DataLoadingCSV.py              // Convert dataframe to label and image path 
│
├── data_split_generator.py         // creates training and testing csv file for model training
│
├── data_bias_generator.py            // creates training and testing csv file for bias checking
|
└── README.md

[3] Steps:

FOR DATASET CREATION:

    There are some steps given below to perform.

    1) Remove class_data folder,if present
    2) run extraction.py to get data from DAiSEE dataset from the path (LOCAL)
    3) Get Angry and Neutral Images from other dataset.
    4) 405 - Train images for each class & 150 - Test images for each class.

*Files to run for Train , evaluate , models.

- For Data Labelling - python LabelledData.py
- For Data Cleaning - python EDA.py
- For Visualization - python EDA.py
- For Data Extraction - python ExtractData.py
- For Facial Emotion detection - python Image_analysis.py (based CNN version)
                               -> image_analysis_var1.py (changed number of con layers)
                               -> image_analysis_var2.py (changed kernel size)
- For Train Model - trained_model.pth 
- For Running Train Model on Dataset or a specific Image - python Application.py
