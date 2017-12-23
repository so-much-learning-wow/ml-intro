import pandas
import numpy as np
from sklearn import metrics

classification = pandas.read_csv('./data/classification.csv')

# 1. Calculate tp, fp, tn, fn
true_positive = len(classification[(classification['true'] == 1) & (classification['pred'] == 1)])
false_positive = len(classification[(classification['true'] == 0) & (classification['pred'] == 1)])
false_negative = len(classification[(classification['true'] == 1) & (classification['pred'] == 0)])
true_negative = len(classification[(classification['true'] == 0) & (classification['pred'] == 0)])

# Save the answer
answer = str(true_positive) + ' ' + str(false_positive) + ' ' + str(false_negative) + ' ' + str(true_negative)
submission_file = open('submissions/metrics/errors_matrix.txt', 'w+')
submission_file.write(answer)
submission_file.close()
print(answer)


# 2. Find basic metrics: precision, recall, F-score, accuracy
accuracy = metrics.accuracy_score(classification['true'], classification['pred'])
precision = metrics.precision_score(classification['true'], classification['pred'])
recall = metrics.recall_score(classification['true'], classification['pred'])
f1_score = metrics.f1_score(classification['true'], classification['pred'])

answer = str(np.round(accuracy, 2)) + ' ' + str(np.round(precision, 2)) + ' ' + str(np.round(recall, 2)) + ' ' + str(np.round(f1_score, 2))
submission_file = open('submissions/metrics/basic_metrics.txt', 'w+')
submission_file.write(answer)
submission_file.close()
print(answer)


# 3. Find the best classifier, based on AUC ROC
scores = pandas.read_csv('./data/scores.csv')
column_names = scores.ix[:, 1:].columns.values.tolist()
auc_rocs = [metrics.roc_auc_score(scores['true'], scores[c_name]) for c_name in column_names]
auc_rocs_with_names = list(zip(auc_rocs, column_names))
auc_rocs_with_names.sort(key = lambda x:x[0])
best_classifier = auc_rocs_with_names[-1][1]

# Save the answer
submission_file = open('submissions/metrics/auc_roc.txt', 'w+')
submission_file.write(best_classifier)
submission_file.close()
print(best_classifier)


# Find the best classifier, based on precision, when recall is more than 70%
pr_curves = [metrics.precision_recall_curve(scores['true'], scores[c_name]) for c_name in column_names]
precisions = list(range(len(pr_curves)))

for index, pr_curve in enumerate(pr_curves):
	# loop through recall thresholds
	i = 0
	while i < len(pr_curve[1]):
		if pr_curve[1][i] <= 0.7:
			break
		else:
			i += 1

	# find the best precision for such recall
	precisions[index] = max(pr_curve[0][:i])

precisions_with_names = list(zip(precisions, column_names))
precisions_with_names.sort(key = lambda x:x[0])
best_classifier = precisions_with_names[-1][1]

# Save the answer
submission_file = open('submissions/metrics/precision.txt', 'w+')
submission_file.write(best_classifier)
submission_file.close()
print(best_classifier)
