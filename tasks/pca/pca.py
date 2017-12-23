import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

prices = pd.read_csv('./data/close_prices.csv')
dj_index = pd.read_csv('./data/djia_index.csv')

# remove first column (it is just dates)
prices = prices.ix[:, 1:]
dj_index = dj_index.ix[:, 1]

pca = PCA(n_components=10)
pca.fit(prices)

# we shoud count, how many features we need to explain 90% of variance
n, s = 0, 0
while s < 0.9:
	s += pca.explained_variance_ratio_[n]
	n += 1

submission_file = open('submissions/90_var.txt', 'w+')
submission_file.write(str(n))
submission_file.close()
print(n)

# magic hacks to get correlation
corr = np.corrcoef(pca.transform(prices).transpose()[0], dj_index)[0][1]
submission_file = open('submissions/correlation.txt', 'w+')
submission_file.write(str(np.round(corr, 2)))
submission_file.close()
print(str(np.round(corr, 2)))


most_influential_company = prices.columns.values[pca.components_[0].tolist().index(max(pca.components_[0]))]
submission_file = open('submissions/most_influential_company.txt', 'w+')
submission_file.write(most_influential_company)
submission_file.close()
print(most_influential_company)
