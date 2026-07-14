import os

import numpy as np

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import typing

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data prep
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region

# set data source and available categories
input_dir = '/home/migalha/code/python_code/AI_projects/parking_classifier/data'
categories = ['busy','free']

data = []
labels = []

#iterating on each picture
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        # get each image, resize them to a small size
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = typing.cast(np.ndarray, resize(img, (15,15))) # just to prevent vscode from freaking out 

        # take the image and append it as a 1D vector and the correct category
        data.append(img.ravel())
        labels.append(category_idx)

# turning lists into nparrays
data = np.asarray(data)
labels= np.asarray(labels)

#endregion

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# making training/testing sets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # input and output sets shuffled and stratified

#endregion

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train classifier
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region

# Support Vector Classification works well in high dimensional spaces ((60x60)x3 = 675 data axes)
classifier = SVC()

parameters = [
        {
        # 'kernel': ['rbf', 'linear'],
        'gamma': [0.01, 0.001, 0.0001], # how much each feature impacts the calibration
        'C': [1, 10, 100, 1000]         # penalty of getting a wrong answer
        }
    ]

# search for the local optimum {gamma, C} gamma[0],C[0] || gamma[0],C[1] || gamma[0],C[2] ||...
grid_search = GridSearchCV(classifier, parameters)

# train
grid_search.fit(x_train, y_train)

#endregion

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# perfomance metrics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region

# select the best training result from the grid 
best_estimator = grid_search.best_estimator_

# run prediction and check it with real label
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(f'y PREDICTION {y_prediction}\ny VALUE {y_test}\nSCORE {score}')

#endregion

# Since classifying parking spots is fairly easy, with this simple model we get 95%+ accuracy