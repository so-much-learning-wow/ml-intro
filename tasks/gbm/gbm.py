import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

def main():
	data = pd.read_csv('./data/gbm-data.csv')
	X, y = data.drop('Activity', axis=1), data['Activity']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

	learning_rate_variants = [1, 0.5, 0.3, 0.2, 0.1]
	train_scores = list(range(5))
	test_scores = list(range(5))

	for i, learning_rate in enumerate(learning_rate_variants):
		train_scores[i], test_scores[i] = learn(learning_rate, X_train, y_train, X_test, y_test)

	# well, you should run the script to be sure in this
	learning_problem = 'overfitting'
	submission_file = open('submissions/learning_problem.txt', 'w+')
	submission_file.write(learning_problem)
	submission_file.close()
	print(learning_problem)

	# now we should analyze, what is the minimum test score log loss for 0.2 learning rate
	min_log_loss = min(test_scores[3])
	min_log_loss_iter = test_scores[3].index(min_log_loss) + 1
	answer = str(np.round(min_log_loss, 2)) + ' ' + str(min_log_loss_iter)
	submission_file = open('submissions/min_log_loss.txt', 'w+')
	submission_file.write(answer)
	submission_file.close()
	print(answer)

	# now we should train random forest classifier and compare results
	model = RandomForestClassifier(random_state=241, n_estimators=min_log_loss_iter)
	model.fit(X_train, y_train)
	predictions = model.predict_proba(X_test)
	# predictions = [x[0] for x in predictions.tolist()] # unpack this stupid format
	# predictions = [1 / (1 + math.exp(-x)) for x in predictions]
	score = log_loss(y_test, predictions)
	answer = str(np.round(score, 2))
	submission_file = open('submissions/rfc_score.txt', 'w+')
	submission_file.write(answer)
	submission_file.close()
	print(answer)

def learn(learning_rate, X_train, y_train, X_test, y_test):
	model = GradientBoostingClassifier(
		n_estimators=250,
		verbose=True,
		random_state=241,
		learning_rate=learning_rate
		)
	model.fit(X_train, y_train)
	
	# plot scores
	test_score = list(range(250))
	train_score = list(range(250))

	for i, predictions in enumerate(model.staged_decision_function(X_test)):
		predictions = [x[0] for x in predictions.tolist()] # unpack this stupid format
		predictions = [1/(1 + math.exp(-x)) for x in predictions]
		test_score[i] = log_loss(y_test, predictions)

	for i, predictions in enumerate(model.staged_decision_function(X_train)):
		predictions = [x[0] for x in predictions.tolist()] # unpack this stupid format
		predictions = [1/(1 + math.exp(-x)) for x in predictions]
		train_score[i] = log_loss(y_train, predictions)

	plt.figure()
	plt.plot(test_score, 'r', linewidth=2)
	plt.plot(train_score, 'g', linewidth=2)
	plt.legend(['test', 'train'])
	plt.show()
	
	return train_score, test_score

if __name__ == '__main__':
	main()