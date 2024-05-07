# importing packages
import numpy as np 
import os
import cv2 as cv
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepate data
input_dir = './clf-data'
categories = ['empty','not_empty']

data = []
labels = []

for idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir,category,file)
        img = cv.imread(img_path)
        resized = cv.resize(img, (15,15))
        data.append(resized.flatten())
        labels.append(idx)

data = np.asarray(data)
labels = np.asarray(labels)


# Training and Test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train classifier
# classifier = SVC()
# parameters = [{'gamma':[0.01,0.001,0.0001], 'C':[1,10,100,1000]}] # SVC(C=10, gamma=0.0001)
# grid_search = GridSearchCV(estimator=classifier, param_grid=parameters)
# grid_search.fit(X_train,y_train)

classifier = RandomForestClassifier()
rf_model = classifier.fit(X_train,y_train)

# Test Performance
# best_estimator = grid_search.best_estimator_
# print(best_estimator)
# y_pred = best_estimator.predict(X_test)

y_pred = rf_model.predict(X_test)

score = accuracy_score(y_test,y_pred)
print(f'{score*100}% of the samples were correctly classified')

#cv.imshow('Image',img)
#cv.waitKey(2000)