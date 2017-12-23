import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

X_train = pd.read_csv('data/features.csv')
X_test = pd.read_csv('data/features_test.csv')
y = X_train['radiant_win']


X_train = X_train.drop([
		'tower_status_radiant',
		'duration',
		'tower_status_dire',
		'radiant_win',
		'barracks_status_dire',
		'barracks_status_radiant'
	], axis=1)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

X_train_scaled = scale(X_train)
cv = KFold(len(X_train), shuffle=True, n_folds=5, random_state=666)
C_variants = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_scores = list(range(len(C_variants)))
for i, C in enumerate(C_variants):
    model = LogisticRegression(penalty='l2', C=C, random_state=666)
    scores = cross_val_score(model, X_train_scaled, y=y, cv=cv, scoring='roc_auc')
    C_scores[i] = scores.mean()
    print('Model created. C value is:', C, 'Score is:', C_scores[i])

best_score = max(C_scores)
best_C = C_variants[ C_scores.index(best_score) ]

X_train = X_train.drop([
        'lobby_type',
        'r1_hero',
        'r2_hero',
        'r3_hero',
        'r4_hero',
        'r5_hero',
        'd1_hero',
        'd2_hero',
        'd3_hero',
        'd4_hero',
        'd5_hero'], axis=1)



