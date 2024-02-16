import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt
from Logistic_regression import *

bc = load_breast_cancer()
X,y = bc.data ,bc.target

X_train, X_test, y_train ,y_test = train_test_split(X,y, test_size=0.2, random_state=42) 

clf = LogisticRegression(lr=0.1)
clf.fit(X_train, y_train)

y_pred =clf.predict(X_test)

print('Predictions:\n',y_pred)
print(f'Accuracy Score: {round(clf.accuracy_score(y_test, y_pred) * 100)}%')




