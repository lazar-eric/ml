import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-data.csv', sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], axis=1))

y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# best = 0

# Uncomment to make, and save the model
# for _ in range(40):
#   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#   model = linear_model.LinearRegression()

#   model.fit(x_train, y_train)

#   accuracy = model.score(x_test, y_test)

#   if (accuracy > best):
#     with open('model.pickle', 'wb') as f:
#       pickle.dump(model, f)

#     best = accuracy

#     print('Accuracy: ', best)

saved_model = open('model.pickle', 'rb')

model = pickle.load(saved_model)

predictions = model.predict(x_test)

for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])

  p = 'G1'

style.use('ggplot')

pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final grade')
pyplot.show()


