import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

data_train = pd.read_csv('./data/salary-train.csv')
data_test = pd.read_csv('./data/salary-test-mini.csv')

encoder = DictVectorizer()
vectorizer = TfidfVectorizer(min_df = 5)
model = Ridge(alpha=1)

# preprocess data
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_train['FullDescription'] = data_train['FullDescription'].str.replace(r'[^a-zA-Z0-9]', ' ')
data_test['FullDescription'] = data_test['FullDescription'].str.replace(r'[^a-zA-Z0-9]', ' ')

# fill missing data
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

# transform features
x_train_desc = vectorizer.fit_transform(data_train['FullDescription'])
x_test_desc = vectorizer.transform(data_test['FullDescription'])

# now we can transform categorical features (after we have filled missing values with nan)
x_train_categ = encoder.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_test_categ = encoder.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

x_train = hstack([x_train_desc, x_train_categ])
x_test = hstack([x_test_desc, x_test_categ])

model.fit(x_train, data_train['SalaryNormalized'])
predictions = model.predict(x_test)

answer = ' '.join(str(np.round(x, 2)) for x in predictions)

submission_file = open('submissions/predictions.txt', 'w+')
submission_file.write(answer)
submission_file.close()

print(answer)
