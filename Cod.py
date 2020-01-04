# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:40:48 2020

@author: Ramya Aravind
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('framingham.csv')

#Finding the columns with missing values
dataset.columns[dataset.isnull().any()]

#Columns with missing values: education, cigsPerDay, BPMeds, totChol, BMI, heartRate, glucose

#Replace BMMeds missing values with 0
dataset['BPMeds'].replace(to_replace = np.nan, value = 0) 

#replace all other missing values with mean of columns
dataset = dataset.apply(lambda x: x.fillna(x.mean()),axis=0)

#X are independent variables & Y is 10 year CHD occurence
X = dataset.iloc[:, 0:15].values
y = dataset.iloc[:, 15].values

#making dummy variables for Education
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#The CODE BELOW DOESNT WORK SO COMMENTED IT
#labelencoder_edu = LabelEncoder()
#X[:,2] = labelencoder_edu.fit_transform(X[:,2])
#onehotencoder = OneHotEncoder(categories=[2])
#X = onehotencoder.fit_transform(X).toarray()

#THIS CODE WORKS TO MAKE DUMMY VARIABLES
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [2])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)


#Splitting dataset into testing and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fit Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#Fit Decision Tree classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train, y_train)

#predicting test set
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Logistic regression classifier accuracy 84.43%
#Decision tree classifier accuracy: 76.23%