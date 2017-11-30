#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys, os
from time import time
from sklearn import tree
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR) + "/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
classifier = tree.DecisionTreeClassifier(min_samples_split=40)

print("no. of features: " + str(len(features_train[0]))) # Number of rows is data points, number of columns is number of features

t0 = time()
classifier.fit(features_train, labels_train)
print("Training time: ", round(time() - t0, 3), "s")

t0 = time()
predictions = classifier.predict(features_test)
print("Prediction time: ", round(time() - t0, 3), "s")

accuracy = classifier.score(features_test, labels_test)
print(accuracy)

#########################################################

# Times & Accuracy w/ min_samples_split = 40
## Number of features: 3785
## Training time: 126.773s
## Prediction time: 0.082s
## Accuracy: 0.977815699659

# Times & Accuracy w/ min_samples_split = 40 & w/ percentile set to only 1 from email_preprocess (1% of the features is used)
## Number of features: 379
## Training time: 7.442s
## Prediction time: 0.002s
## Accuracy: 0.966439135381