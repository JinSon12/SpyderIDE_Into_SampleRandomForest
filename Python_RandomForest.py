#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on WED Mar  4 12:18:18 2020

@author: jinson
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


iris = datasets.load_iris() 

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
sns.pairplot(df, hue='species')

# to find more about a function, command + click 
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], 
                                                    iris.target, 
                                                    test_size=0.25, 
                                                    stratify=iris.target, 
                                                    random_state=123456)

# RandomForestClassifier - python is a case sensitive language, so "T"rue always
rf = RandomForestClassifier(n_estimators=100, 
                            oob_score=True, 
                            random_state=123456)
rf.fit(X_train, y_train)


# With the model we created, let's classify the "test" set (accuracy_score)
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)


# prepended r' or f' and {variable} or {number} = formatting method. 
# (so f'xxxx {xx:.3}) = display up to 3 decimal digits 
print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')


# Using confusion_matrix to take a further look into the model 
# (and be more confident in our model)
# you can view this on the Variable Explorer and from IPython Console 
# tip : inline - x interactive, auto : interactive

cm = pd.DataFrame(confusion_matrix(y_test, predicted), 
                  columns=iris.target_names, 
                  index=iris.target_names)
sns.heatmap(cm, annot=True)


