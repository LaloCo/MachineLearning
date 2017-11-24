#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
from time import time
#sys.path.append("../tools/")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR) + "/tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
t0 = time()
classifier.fit(features_train, labels_train)
print("Training time: ", round(time() - t0, 3), "s")
GaussianNB(priors=None)
t0 = time()
predictions = classifier.predict(features_test)
print("Prediction time: ", round(time() - t0, 3), "s")

accuracy = classifier.score(features_test, labels_test)
print(accuracy)

#########################################################

# Naive Bayes times
## Training time: 1.961s
## Prediction time: 0.286s