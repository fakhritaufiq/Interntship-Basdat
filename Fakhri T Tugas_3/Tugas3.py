# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:19:47 2019

@author: fakhri
"""

import pandas as pd

df = pd.read_csv('insurance.csv')

# Pisahkan Independent dan Dependent variable
X = df.iloc[0:99, :-1].values
Y = df.iloc[0:99, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#Encode Sex
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encode Smoker
labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])
#Encode Region
labelencoder_X_3 = LabelEncoder()
X[:, 5] = labelencoder_X_3.fit_transform(X[:, 5])

onehotencoder = OneHotEncoder(categorical_features = [1, 4, 5])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# Train Linear Regression
regressor.fit(X_train, Y_train)

# Prediksi
Y_pred = regressor.predict(X_test)

#Melihat score
from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)
print(score)