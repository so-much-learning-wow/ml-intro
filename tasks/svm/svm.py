import pandas
import numpy
from sklearn.svm import SVC

data = pandas.read_csv('./data/svm.csv', header=None)
objects = data.ix[:, 1:]
classes = data.ix[:, 0]

model = SVC(kernel='linear', C=100000, random_state=241)
model.fit(objects, classes)

answer = ' '.join(str(x + 1) for x in model.support_)

submission_file = open('submissions/svm/support_vectors.txt', 'w+')
submission_file.write(answer)
submission_file.close()

print(answer)
