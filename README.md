# MNIST-LinearClassifier-Python-No_Packages
MNIST Linear Classification (OvA) multi-class classification implemented in Python, no packages

This is a short project on classification using the MNIST dataset. This project involves implimenting One vs All (OvA) multi-class linear classification in Python without the use of machine learning packages. It helps to give an overall view of how a simple multi-class linear classifier works.

Key concepts - MNIST, OvA Multi-class Classification, Python, no packages

Data was downloaded from the Kaggle Digit Recongnizer
https://www.kaggle.com/c/digit-recognizer/data

Usage
+ MNIST_LinearClassification_OvA.py = main Python script. 
+ data - zipped data file
  + train.csv = training data (42000 images with 745 features, all labelled 0-9)
  + test.csv = testing data (28000 images with 745 features, no labels)

Settings: 2000 iterations per class, alpha = 0.0001

Total training time ~ 8 min.

Results: Accuracy on testing data (calculated by Kaggle) = 91.1%
A reasonable accuracy for this exercise, shows the effectiveness of simple linear classification of multi-class data
