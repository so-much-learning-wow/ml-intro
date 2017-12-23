import pandas
import numpy
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

train_data = pandas.read_csv('./data/perceptron-train.csv', header=None)
test_data = pandas.read_csv('./data/perceptron-test.csv', header=None)

x_train = train_data.ix[:, 1:]
x_test = test_data.ix[:, 1:]

y_train = train_data.ix[:, 0]
y_test = test_data.ix[:, 0]

x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)

model = Perceptron(random_state = 241)
model.fit(x_train, y = y_train)
predictions = model.predict(x_test)
accuracy_without_scale = accuracy_score(y_test, predictions)

model = Perceptron(random_state = 241)
model.fit(x_train_scaled, y = y_train)
predictions = model.predict(x_test_scaled)
accuracy_with_scale = accuracy_score(y_test, predictions)

accuracy_imrovement = accuracy_with_scale - accuracy_without_scale

submission_file = open('submissions/perceptron/accuracy_imrovement.txt', 'w+')
submission_file.write(str( numpy.round(accuracy_imrovement, 2) ))
submission_file.close()

print( str( numpy.round(accuracy_imrovement, 2) ) )
