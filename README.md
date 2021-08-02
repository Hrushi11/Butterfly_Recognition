# Butterfly Recognition

The datset for this project is taken from - [Butterfly Image Classification](https://www.kaggle.com/gpiosenka/butterfly-images40-species). <br>
The test accuracy recieved was **96.70%**

Train, Test. Validation data set for 50 butterfly species. All images are 224 X 224 X 3 in jpg format .
Train set consists of 4955 images partitioned into 50 sub directories one for each species.
Test set consists of 250 images partitioned into 50 sub directories with 5 test images per species.
Valid set consists of 250 images partitioned into 50 sub directories with 5 validation images per species.
1 CSV file is included with 3 columns filepaths, labels and dataset. filepaths is the relative path to an image.
labels is the string name of the species associate with the image file.
dataset species the directory (train, test, valid) that contains the image file.

Model 2 is the best model and is the one used in the app.

**Random images from the dataset :**

![IMG](https://github.com/Hrushi11/Butterfly_Recognition/blob/main/images/random%20images.png?raw=true)

**Predictions made by the model :**

![IMG](https://github.com/Hrushi11/Butterfly_Recognition/blob/main/images/predictions1.jpeg?raw=true)

**Wrongly predicted images :**

![IMG](https://github.com/Hrushi11/Butterfly_Recognition/blob/main/images/wrong%20images.png?raw=true)
