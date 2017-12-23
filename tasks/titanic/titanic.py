import pandas
import numpy as np

data = pandas.read_csv('./data/titanic.csv', index_col='PassengerId')

# task 1 - amount of men and women there were
sex_value_counts = data['Sex'].value_counts()
answer = str(sex_value_counts['male']) + " " + str(sex_value_counts['female'])

submission_file = open('submissions/titanic/sex.txt', 'w+')
submission_file.write(answer)
submission_file.close()

print('sex: ' + answer)

# task 2 - percent of survived
survived_counts = data['Survived'].value_counts()
survived_portion = float(survived_counts[1]) / float(len(data))
survived_pecent = np.round((100 * survived_portion), 2)

submission_file = open('submissions/titanic/survived.txt', 'w+')
submission_file.write(str(survived_pecent))
submission_file.close()

print('survived: ' + str(survived_pecent))

# task 3 - percent of first-class passengers
pclass_counts = data['Pclass'].value_counts()
first_class_portion = float(pclass_counts[1]) / float(len(data))
first_class_pecent = np.round((100 * first_class_portion), 2)

submission_file = open('submissions/titanic/1_class.txt', 'w+')
submission_file.write(str(first_class_pecent))
submission_file.close()

print('first_class: ' + str(first_class_pecent))

# task 4 - mean and median of passengers' age
age_mean = np.round((data['Age'].mean()), 2)
age_median = np.round((data['Age'].median()), 2)
answer = str(age_mean) + " " + str(age_median)

submission_file = open('submissions/titanic/age.txt', 'w+')
submission_file.write(answer)
submission_file.close()

print('age: ' + answer)

# task 5 - pearson correlation between SibSp and Parch
correlation = np.round(data['Parch'].corr(data['SibSp']), 2)

submission_file = open('submissions/titanic/corr.txt', 'w+')
submission_file.write(str(correlation))
submission_file.close()

print('correlation: ' + str(correlation))

# task 6 - most popular female name
data['FirstName'] = data['Name'].apply(lambda x: x[x.find('. ') + 2 : ][ : x.find(' ') ] )
women = data.loc[data['Sex'] == 'female']
most_popular_female_name = women['FirstName'].mode()[0]

submission_file = open('submissions/titanic/female_name.txt', 'w+')
submission_file.write(most_popular_female_name)
submission_file.close()

print('female name: ' + most_popular_female_name)

