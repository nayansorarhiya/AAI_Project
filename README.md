# AAI_Project

1. File Enumeration and Descriptions:

In our Project we have many files which are important and listed below.

• New Date folder contains all the data required for the project. Under that folder we have 2 various category "Test" and "Train" which can be used for testing and training purpose. Under both of them we have 4 more folders names as Neautral , Angry , Boredom , engagement which represents the categories.

- 'ExtractData.py' - we are using 2 database, one contains images for neautral and angry emotion but the other 2 category are taken from DaiSEE dataset which contains vedios of 15GB so this file will read the label CSV file for this specific dataset and select the vedios according to the emotion and take a frames from it and create a new dataset for us but the DB is too larga so we are not able to upload it on a github but it is working for our system.
- 'LabelledData.py' - This file iterate through all the images from the dataset named newData Folder and create a CSv file named (image_info.csv) which is our labelled CSV file. This CSV file contains information such as image name , emotion  , imagee format (JPG , PNG) , Pixel and size.
- 'EDA.py' -
    [1] def convert_all_png() - This converts all the pngs to jpg. In this process, it will use sub method " def convert_png_to_jpg(directory) ".
    [2] def resize_all_images(param1  , param2) - This Method will rezise the images in the size param1 X param2.
    [3] def barchartforImagesize() - This method is used to show the different sizes of images.
    [4] def intensity_for_random() - This method is used to show the RGB intensity for 25 random images.
    [5] def plotbarChart(data_type) - This is the bar chart which shows the number of images for all the 4 emotions in both train and test dataset. In the data_type, we can pass "Train" or "Test" according to the dataset.
    [6] def plotRandom() - This method is used to plot 25 random images in 5 * 5 grid.


2. Data Cleaning (a):

• Environment Setup:
  - Ensure you have Python 3.11 or higher installed.
  - Install required dependencies
    (opencv-python
     pandas
     Pillow
     matplotlib
     tensorflow
     scipy
     numpy )
• Data Cleaning and processing:
  - Data Cleaning is done by one time code execution of data cleaning and processing code provided in EDA.py file and in our generated CSV file we found no missing data.

3. Data Visualization:
   - EDA.py contains all the relative coding for EDA which mainly includes Class ditsribution trough plotting various graphs , label cheking for random images , pixel intensity distrubution graph for random images. Generated graphs are provided in the report.
   - Here visulization of data is done by using some libraries like Matplotlib , tensorflow , pillow , etc.



- - The flow to run project - -
  - first we have to run ExtractData.py becuase it is taking frames out of video for bordom and engagement. The video dataset is larger so we have not provided it here so this file will not run in your system.
  - Then we have to run LabelledData.py file for labelling data which will create image_info.csv.
  - Then EDA.py file is main file of our project which will do pre-processing , data cleaning  ,  and visualization.

**The first 2 files should be run only once and we have already run it so simply by running EDA.py file, you can see the output.**
