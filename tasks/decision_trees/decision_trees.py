import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data = pandas.read_csv('./data/titanic.csv', index_col='PassengerId')

# we work only with specific labels (the task prescribes this)
objects = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

# remove objects, containing NaN values in some labels
objects = objects.dropna()

# getting our target values
classes = objects['Survived']

# now we can drop target column from objects
objects = objects.drop('Survived', axis=1)

# convert string features to numbers
objects['Sex'] = objects['Sex'].apply(lambda x: 0 if x == 'female' else 1)

model = DecisionTreeClassifier(random_state=241)
model.fit(objects, classes)

importances = model.feature_importances_
importances_names_and_values = list(zip(importances, list(objects.dtypes.index)))
importances_names_and_values.sort()

sorted_importances_names = [name for val, name in importances_names_and_values]

answer = sorted_importances_names[-1] + " " + sorted_importances_names[-2]

submission_file = open('submissions/decision_trees/important_features.txt', 'w+')
submission_file.write(answer)
submission_file.close()

print(answer)

