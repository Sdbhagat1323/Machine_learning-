# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature Scaling
""""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(Y_train)
""""
# simple linear regrassion modelto the training set 
from sklearn.linear_model import LinearRegression 
regressar = LinearRegression()
regressar.fit(X_train, Y_train)

# predicting the test set reuslts
Y_pred = regressar.predict(X_test)

# visualization of the training set 
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressar.predict(X_train))
plt.title("salary vs Experience (Training Set)")
plt.xlabel("Year of Experience")
plt.ylabel("salary")
plt.show()


#visualization of the test set 
plt.scatter(X_test, Y_test, color = "blue")
plt.plot(X_train, regressar.predict(X_train))
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Year of Experience")
plt.ylabel("salary")
plt.show()








