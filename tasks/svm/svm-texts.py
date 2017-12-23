import pandas
import numpy
import math
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV

news_group = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()
news_group.data = vectorizer.fit_transform(news_group.data)

# initialize valitador
cv = KFold(len(news_group.target), shuffle=True, n_folds=5, random_state=241)

# declare parameters
C_vals = [10**x for x in range(-5, 6)]
parameters = { 'kernel': ['linear'], 'C': C_vals, 'random_state': [241] }

svc = SVC(random_state=241)

model = GridSearchCV(svc, parameters, cv=cv)
model.fit(news_group.data, news_group.target)
best_params = model.best_params_
C = best_params['C']

model = SVC(kernel='linear', C=C, random_state=241)
model.fit(news_group.data, news_group.target)

# gathering answer
feature_names = vectorizer.get_feature_names()
features_weights = model.coef_.toarray().tolist()[0]
features_weights = [math.fabs(w) for w in features_weights]
feature_weights_and_names = list(zip(features_weights, feature_names))
feature_weights_and_names.sort(key=lambda f: f[0])
top_words = [f[1] for f in feature_weights_and_names[-10:]]
top_words.sort()

answer = ','.join(top_words)

submission_file = open('submissions/svm-texts/top_words.txt', 'w+')
submission_file.write(answer)
submission_file.close()

print(answer)
