import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

### Read in data and display first 5 rows
features = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\B5_asc.csv")
labels = np.array(features['Capacity'])

### Saving feature names for later use
feature_list = list(features.columns)

### Convert to numpy array
features = np.array(features)


### Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#to sort the train_features based on the ascending order of capacity
train_features = train_features[np.argsort(train_features[:, 10])]
#delete the last column of train_features which is the capacity column
train_features = np.delete(train_features, 10, 1)
train_labels.sort()
#to sort the test_features based on the ascending order of capacity
test_features = test_features[np.argsort(test_features[:, 10])]
#delete the last column of test_features which is the capacity column
test_features = np.delete(test_features, 10, 1)
test_labels.sort()


from elm import ELMRegressor
#from random_hidden_layer import RBFRandomHiddenLayer,SimpleRandomHiddenLayer
#from elm.py import ELMRegressor
elm = ELMRegressor(n_hidden=30)
elm.fit(train_features, train_labels)
#e.fit(train_features, train_labels)
#train_res.append(e.score(train_features, train_labels))
#test_res.append(e.score(test_features, test_labels))

total = 0
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = ELMRegressor(n_hidden=30)
    model.fit(train_features, train_labels)  #fit the model
    pred=model.predict(test_features) #make prediction on test set
    error = sqrt(mean_squared_error(test_labels,pred)) #calculate rmse
    total = total + error
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
avg = total / K
print('Average RMSE value for k =', avg, )

#pred=elm.predict(test_features)

#accuracy matrix
#mse = mean_squared_error(test_labels, pred)
#print("Mean Squared Error:",mse)
###Accuracy metric
#rmse = math.sqrt(mse)
#print("Root Mean Squared Error:", rmse)

#mae = mean_absolute_error(test_labels, pred)
#print("Mean Absolute Error:", mae)
