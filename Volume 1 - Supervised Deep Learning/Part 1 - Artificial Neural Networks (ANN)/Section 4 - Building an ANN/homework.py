# -*- coding: utf-8 -*-

#Artificial Neural Network

#Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# preprocessing data
item = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
item[:, 1] = LabelEncoder().fit_transform(item[:, 1])
item[:, 2] = LabelEncoder().fit_transform(item[:, 2])

#dummy variables
item = onehotencoder.transform(item).toarray()
item = item[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
item = sc.transform(item)



# part 2 - Now let's make the ANN

#Importing the Keras libraries and package
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

#part 3 - Making the predictions and evaluating the model

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(item)
y_pred = (y_pred > 0.5)

#y_pred is false. It means that a customer will not exit the bank, so we should not leave the customer.

#So should we say goodbye to that customer ?
# no.