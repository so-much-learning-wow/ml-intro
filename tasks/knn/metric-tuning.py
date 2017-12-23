import numpy
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

boston = load_boston()
boston.data = scale(boston.data)

possible_p_vals = numpy.linspace(1, 10, num=200)
scores = list(range(200))

# we must use cross validation
cv = KFold(len(boston.target), shuffle=True, n_folds=5, random_state=42)

for i, p in enumerate(possible_p_vals):
	current_model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
	current_scores = cross_val_score(current_model, boston.data, y=boston.target, cv=cv, scoring='mean_squared_error')
	scores[i] = current_scores.mean()
	print(scores[i])

best_score = max(scores)
best_p = possible_p_vals[ scores.index(best_score) ]

submission_file = open('submissions/knn-metric-tuning/best_p.txt', 'w+')
submission_file.write(str( numpy.round(best_p, 2) ))
submission_file.close()

print( numpy.round(best_p, 2) )
