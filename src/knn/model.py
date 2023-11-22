import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))

y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# best = 0

# for k in range(14):
#   if (k == 0): continue

#   model = KNeighborsClassifier(n_neighbors=k)

#   model.fit(x_train, y_train)

#   accuracy = model.score(x_test, y_test)

#   if (accuracy > best):
#     with open('model.pickle', 'wb') as f:
#       pickle.dump(model, f)

#     best = accuracy

#     print('Accuracy: ', best, ' for k: ', k)

saved_model = open('model.pickle', 'rb')

model = pickle.load(saved_model)

predictions = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predictions)):
    print("Predictions: ", names[predictions[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
