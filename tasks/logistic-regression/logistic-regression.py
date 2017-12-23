import pandas
import math
import numpy
from sklearn.metrics import roc_auc_score

def main():
	data = pandas.read_csv('./data/logistic-regression.csv', header=None)
	x = data.ix[:, 1:].as_matrix()
	y = data.ix[:, 0]
	
	weights_without_reg = gradient(x, y)
	weights_with_reg = gradient(x, y, 10)

	# print(weights_without_reg)
	# print(weights_with_reg)

	predictions_without_reg = [x[:,0][i] * weights_without_reg[0] + x[:,1][i] * weights_without_reg[1] for i, v in enumerate(y) ]
	predictions_with_reg = [x[:,0][i] * weights_with_reg[0] + x[:,1][i] * weights_with_reg[1] for i, v in enumerate(y) ]

	score_without_reg = roc_auc_score(y, predictions_without_reg)
	score_with_reg = roc_auc_score(y, predictions_with_reg)

	answer = str(numpy.round(score_without_reg, 3)) + " " + str(numpy.round(score_with_reg, 3))

	submission_file = open('submissions/logistic-regression/scores.txt', 'w+')
	submission_file.write(answer)
	submission_file.close()

	print(answer)



def gradient(x, y, regularization_coef = 0):
	learning_rate = 0.1
	convergence_criteria = 1e-5
	max_iterations_left = 10**5
	current_distance = float("inf")
	w_1 = 0
	w_2 = 0
	l = len(y)

	while current_distance > convergence_criteria and max_iterations_left > 0:
		tmp_w_1, tmp_w_2 = w_1, w_2

		w_1 = w_1 + (learning_rate/l) * sum( [y[i] * x[:,0][i] * (1 - 1/( 1 + math.exp( -y[i] * ( tmp_w_1 * x[:,0][i] + tmp_w_2 * x[:,1][i]) ) ) ) for i, v in enumerate(y)] ) - learning_rate * regularization_coef * w_1
		w_2 = w_2 + (learning_rate/l) * sum( [y[i] * x[:,1][i] * (1 - 1/( 1 + math.exp( -y[i] * ( tmp_w_1 * x[:,0][i] + tmp_w_2 * x[:,1][i]) ) ) ) for i, v in enumerate(y)] ) - learning_rate * regularization_coef * w_2

		max_iterations_left -= 1
		current_distance = math.sqrt( (w_1 - tmp_w_1)**2 + (w_2 - tmp_w_2)**2 )
		# print(current_distance)

	return (w_1, w_2)

if __name__ == '__main__':
	main()