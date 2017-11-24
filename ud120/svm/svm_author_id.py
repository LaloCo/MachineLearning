#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from sklearn.svm import SVC
import sys, os
from time import time
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR) + "/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

## Slice the data to only 1%
# features_train = features_train[:len(features_train)//100] # / makes a float, // makes an int
# labels_train = labels_train[:len(labels_train)//100]

classifier = SVC(kernel='rbf', C=10000.0)
t0 = time()
classifier.fit(features_train, labels_train)
print("Training time: ", round(time() - t0, 3), "s")

t0 = time()
predictions = classifier.predict(features_test)
# Printing predictions
## print("0 for Sara, 1 for Chris")
## print(predictions[10])
## print(predictions[26])
## print(predictions[50])

# Printing Chris Predictions
print("Prediction time: ", round(time() - t0, 3), "s")
print("Predictions for Chris:")
predictionList = list(predictions)
print(predictionList.count(1))
print("Predictions for Sara:")
print(predictionList.count(0))

accuracy = classifier.score(features_test, labels_test)
print(accuracy)

#########################################################

# Times & Accuracy
## Training time: 399.108s
## Prediction time: 41.689s
## Accuracy: 0.984072810011

# Times & Accuracy with 1% of the data to train, 100% of the data to test
## Training time: 0.245s
## Prediction time: 1.999s
## Accuracy: 0.884527872582

# Times & Accuracy with 1% of the data to train, 100% of the data to test & rbg (instead of linear) kernel
## Training time: 0.251s
## Prediction time: 2.252s
## Accuracy: 0.616040955631

# Times & Accuracy with 1% of the data to train, 100% of the data to test & rbg (instead of linear) kernel & C = 10.0
## Training time: 0.308s
## Prediction time: 2.2s
## Accuracy: 0.616040955631

# Times & Accuracy with 1% of the data to train, 100% of the data to test & rbg (instead of linear) kernel & C = 100.0
## Training time: 0.19s
## Prediction time: 2.103s
## Accuracy: 0.616040955631

# Times & Accuracy with 1% of the data to train, 100% of the data to test & rbg (instead of linear) kernel & C = 1000.0
## Training time: 0.21s
## Prediction time: 2.902s
## Accuracy: 0.821387940842

# Times & Accuracy with 1% of the data to train, 100% of the data to test & rbg (instead of linear) kernel & C = 10000.0
## Training time: 0.214s
## Prediction time: 1.784s
## Accuracy: 0.892491467577

# Times & Accuracy with rbg (instead of linear) kernel & C = 10000.0
## Training time: 224.801s
## Prediction time: 22.611s
## Accuracy: 0.990898748578