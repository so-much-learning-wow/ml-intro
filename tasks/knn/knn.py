import pandas
import numpy
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

def main():
	data = pandas.read_csv('./data/wine.csv', header=None)
	classes = data.ix[:, 0]
	features = data.ix[:, 1:]
	scaled_features = scale(features)

	# first, calculate errors and accuracy without scaling
	n_neighbors, score = calculate_k_and_errors(features, classes)

	submission_file = open('submissions/knn/k_without_scale.txt', 'w+')
	submission_file.write(str(n_neighbors))
	submission_file.close()

	submission_file = open('submissions/knn/score_without_scale.txt', 'w+')
	submission_file.write(str( numpy.round(score, 2) ))
	submission_file.close()

	print(n_neighbors, score)

	# now we should calculate errors and accuracy with scaled features
	n_neighbors, score = calculate_k_and_errors(scaled_features, classes)

	submission_file = open('submissions/knn/k_with_scale.txt', 'w+')
	submission_file.write(str(n_neighbors))
	submission_file.close()

	submission_file = open('submissions/knn/score_with_scale.txt', 'w+')
	submission_file.write(str( numpy.round(score, 2) ))
	submission_file.close()

	print(n_neighbors, score)

def calculate_k_and_errors(X, y):

	# we must calculate best k between 1 and 50
	possible_neighbors_amount = list(range(1, 51))
	scores = list(range(1, 51))

	# we must use cross validation
	cv = KFold(len(y), shuffle=True, n_folds=5, random_state=42)

	# calculating optimal k and scores
	for i, n_neighbors in enumerate(possible_neighbors_amount):
		current_model = KNeighborsClassifier(n_neighbors = n_neighbors)
		current_scores = cross_val_score(current_model, X, y=y, cv=cv)
		scores[i] = current_scores.mean()

	score = max(scores)
	n_neighbors = scores.index(score) + 1

	return n_neighbors, score

if __name__ == '__main__':
	main()