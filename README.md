# Emotion-Recognition
This project recognizes human faces and their corresponding emotions from a webcam feed. Used OpenCV and Deep Learning. The model I have used is taken from here https://github.com/oarriaga/face_classification

![picture alt](https://github.com/anwarshaikh078/Emotion-Recognition/blob/master/gif/abc.gif)

## Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.
* tensorflow 
* numpy
* scipy
* opencv-python
* pillow
* pandas
* matplotlib
* h5py
* keras

## How to use code
After installing all the dependencies, follow the below steps

Step 1: Clone or download this repository
  git clone https://github.com/anwarshaikh078/Emotion-Recognition/
  
Step 2: Get the dataset ie fer2013.tar.gz from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

Step 3: Move the downloaded file to the datasets directory inside this repository. Extract the fer2013.tar.gz

Step 4: Run model.py file

## Deep Learning Model
The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.
![picture alt](https://github.com/anwarshaikh078/Emotion-Recognition/blob/master/modelimg.JPG)

Credit
* Computer vision powered by OpenCV.
* Neural network scaffolding powered by Keras with Tensorflow.
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).
