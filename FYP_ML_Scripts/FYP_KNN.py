import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


### Read in data and display first 5 rows
features = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\Input_Data\\B6_US.csv")

### Labels are the values we want to predict, 'actual' is the example's target to predict
labels = np.array(features['Capacity'])

### Saving feature names for later use
feature_list = list(features.columns)

### Convert to numpy array
features = np.array(features)


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


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
#train_labels = sc.fit_transform(train_labels)
test_features = sc.transform (test_features)
#test_labels = sc.transform (test_labels)

from sklearn.decomposition import PCA
pca = PCA(n_components= 6)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)

#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

total = 0
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K, algorithm = 'auto')

    model.fit(train_features, train_labels)  #fit the model
    pred=model.predict(test_features) #make prediction on test set
    error = sqrt(mean_squared_error(test_labels,pred)) #calculate rmse
    total = total + error
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

avg = total / K
print('Average RMSE value for k =', avg, )

#plotting the rmse values against k values
#curve = pd.DataFrame(rmse_val) #elbow curve
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Testing Accuracy')
#plt.plot(rmse_val)
#plt.show()


#Set the new K based on what was determined to be the best from above
#knn = neighbors.KNeighborsRegressor(n_neighbors = 5)
#knn.fit(train_features,train_labels)

#pred = knn.predict(test_features)
#print(y_predict)

#mse = mean_squared_error(test_labels, pred)
#print("Mean Squared Error:",mse)
###Accuracy metric
#rmse = math.sqrt(mse)
#print("Root Mean Squared Error:", rmse)

#mae = mean_absolute_error(test_labels, pred)
#print("Mean Absolute Error:", mae)

#plt.xlabel('Re Resistance')
#plt.ylabel('Capacity')
#plt.plot(test_features, knn.predict(test_features), color = 'blue')
#plt.show()
