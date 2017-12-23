import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold, cross_val_score

data = pd.read_csv('./data/abalone.csv')

# transform Sex column to numbers
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.drop('Rings', axis=1)
y = data['Rings']

cv = KFold(len(y), shuffle=True, n_folds=5, random_state=1)
n_estimators_variants = list(range(1, 51))
n_estimators_scores = list(range(1, 51))

for i, n_estimators in enumerate(n_estimators_variants):
	model = RandomForestRegressor(n_estimators=n_estimators, random_state=1)
	scores = cross_val_score(model, X, y=y, cv=cv, scoring='r2')
	n_estimators_scores[i] = scores.mean()
	print(n_estimators_scores[i])
	

# we shoud calculate, how many trees we need to get r2 more than 0.52
n_trees_index = [ i for i, score in enumerate(n_estimators_scores) if score > 0.52 ][0]
n_trees = n_estimators_variants[ n_trees_index ]

submission_file = open('submissions/n_trees.txt', 'w+')
submission_file.write(str(n_trees))
submission_file.close()
print(n_trees)
