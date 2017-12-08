#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import os
import pickle
from sklearn.linear_model import LinearRegression
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR) + "/tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open(os.path.dirname(CURRENT_DIR) + "/final_project/final_project_dataset_modified_unix.pkl", "rb") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = os.path.dirname(CURRENT_DIR) + '/tools/python2_lesson06_keys_unix.pkl')
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
reg = LinearRegression()
reg.fit(feature_train, target_train)
slope = reg.coef_
intercept = reg.intercept_
# score = reg.score(feature_train, target_train) # Quiz where same training data is used for testing
score = reg.score(feature_test, target_test) 
print("Trained with train data:")
print(slope)
print(intercept)
print(score)

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
    reg.fit(feature_test, target_test)
    plt.plot(feature_train, reg.predict(feature_train), color="b") 
    slope = reg.coef_
    intercept = reg.intercept_
    score = reg.score(feature_test, target_test) 
    print("Trained with test data:")
    print(slope)
    print(intercept)
    print(score)
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()