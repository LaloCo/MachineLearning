#!/usr/bin/python

import pickle
import os
import sys
import matplotlib.pyplot
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR) + "/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dict = pickle.load( open(CURRENT_DIR + "\\../final_project/final_project_dataset_unix.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()