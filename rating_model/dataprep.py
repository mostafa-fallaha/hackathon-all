import pandas as pd


data = pd.read_csv('../../apps/123.csv')
data = pd.get_dummies(data, columns=['Android Ver'])
X = data.drop(['App', 'Rating', 'Category', 'Last Updated'], axis=1)
y = data['Rating'].values
print(X.columns)