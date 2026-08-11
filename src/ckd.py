# -*- coding: utf-8 -*-
"""CKD.ipynb
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/content/kidney_disease.csv")
data.head(10)

data.columns

"""dropping id column and removing duplicate values as well from every attribute"""

for i in data.drop("id",axis=1).columns:
    print('unique values in "{}":\n'.format(i),data[i].unique())

data.info()

data.describe()

"""#Data Cleaning"""

for i in range(data.shape[0]):
    if data.iloc[i,25]=='ckd\t':
        data.iloc[i,25]='ckd'
    if data.iloc[i,20] in [' yes','\tyes']:
        data.iloc[i,20]='yes'
    if data.iloc[i,20]=='\tno':
        data.iloc[i,20]='no'
    if data.iloc[i,21]=='\tno':
        data.iloc[i,21]='no'
    if data.iloc[i,16]=='\t?':
        data.iloc[i,16]=np.nan
    if data.iloc[i,16]=='\t43':
        data.iloc[i,16]='43'
    if data.iloc[i,17]=='\t?':
        data.iloc[i,17]=np.nan
    if data.iloc[i,17]=='\t6200':
        data.iloc[i,17]= '6200'
    if data.iloc[i,17]=='\t8400':
        data.iloc[i,17]= '6200'
    if data.iloc[i,18]=='\t?':
        data.iloc[i,18]=np.nan
    if data.iloc[i,25]=='ckd':
        data.iloc[i,25]='yes'
    if data.iloc[i,25]=='notckd':
        data.iloc[i,25]='no'

data.shape

data = pd.get_dummies(data)
data

data.shape

#separate features and labels
labels = np.array(data['classification_yes'])
features = data.drop(['id', 'classification_no', 'classification_yes'], axis=1)  # Exclude unnecessary columns
features = np.array(features)

features.shape

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with 80% for training and 20% for testing
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, random_state=42)

import numpy as np

# Find rows without NaN values in any column and keep them
mask = ~np.isnan(train_features).any(axis=1)
train_features = train_features[mask]
train_labels = train_labels[mask]

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)

import numpy as np

# Check for NaN or Infinity values in test_features
print(np.isnan(test_features).any())
print(np.isfinite(test_features).all())

from sklearn.impute import SimpleImputer

# Create a SimpleImputer to fill NaN values with the mean of the column
imputer = SimpleImputer(strategy='mean')

# Fit and transform the test_features array using the imputer
test_features = imputer.fit_transform(test_features)

prediction = rf.predict(test_features)

prediction

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test_labels, prediction)
print(f"Mean Squared Error: {mse}")

errors = abs(prediction-test_labels)

errors

mae = (errors/test_labels)

mae

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

threshold = 0.5
binary_predictions = (prediction > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(test_labels, binary_predictions)

# Calculate precision, recall, and F1-score
precision = precision_score(test_labels, binary_predictions)
recall = recall_score(test_labels, binary_predictions)
f1 = f1_score(test_labels, binary_predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

accuracy

from sklearn.tree import export_graphviz
import pydot

tree = rf.estimators_[10]

export_graphviz(tree, out_file='tree.dot')
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
