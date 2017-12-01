#!/usr/bin/python

import matplotlib.pyplot as plt
from time import time
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
t0 = time()
clf.fit(features_train, labels_train)
print("Training time: ", round(time() - t0, 3), "s")

t0 = time()
predictions = clf.predict(features_test)
print("Prediction time: ", round(time() - t0, 3), "s")

accuracy = clf.score(features_test, labels_test)
print(accuracy)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

#########################################################

# Times & Accuracy with n_neighbors = 3 & weight = 'uniform'
## Training time: 0.002s
## Prediction time: 0.002s
## Accuracy: 0.936

# Times & Accuracy with n_neighbors = 5 & weight = 'uniform'
## Training time: 0.002s
## Prediction time: 0.002s
## Accuracy: 0.92

# Times & Accuracy with n_neighbors = 1 & weight = 'uniform'
## Training time: 0.002s
## Prediction time: 0.007s
## Accuracy: 0.94

# Times & Accuracy with n_neighbors = 1 & weight = 'distance'
## Training time: 0.002s
## Prediction time: 0.002s
## Accuracy: 0.94

# Times & Accuracy with n_neighbors = 3 & weight = 'distance'
## Training time: 0.003s
## Prediction time: 0.003s
## Accuracy: 0.936

# Times & Accuracy with n_neighbors = 1 & weight = 'distance' & algorithm = 'auto'
## Training time: 0.002s
## Prediction time: 0.003s
## Accuracy: 0.94

# Times & Accuracy with n_neighbors = 1 & weight = 'distance' & algorithm = 'ball_tree'
## Training time: 0.005s
## Prediction time: 0.003s
## Accuracy: 0.94

# Times & Accuracy with n_neighbors = 1 & weight = 'distance' & algorithm = 'kd_tree'
## Training time: 0.004s
## Prediction time: 0.002s
## Accuracy: 0.94

# Times & Accuracy with n_neighbors = 1 & weight = 'distance' & algorithm = 'brute'
## Training time: 0.001s
## Prediction time: 0.019s
## Accuracy: 0.94