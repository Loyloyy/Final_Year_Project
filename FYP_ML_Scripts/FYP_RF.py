import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

### Read in data and display first 5 rows
features = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\Input_Data\\B5_US.csv")
#print (features.head(5))

###Identify Anomalies/ Missing Data, shape of features will show how many rows and colums the dataset has
#print('The shape of our features is:', features.shape)

### Descriptive statistics for each column
###using graph can be easier to spot anomalies
#print (features.describe())

### Labels are the values we want to predict, 'actual' is the example's target to predict
labels = np.array(features['Capacity'])

### Remove the labels from the features
### axis 1 refers to the columns
#features= features.drop('Capacity', axis = 1)

### Saving feature names for later use
feature_list = list(features.columns)

### Convert to numpy array
features = np.array(features)


### Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
    test_size = 0.25, random_state = 40)
#print (train_features)
#to sort the train_features based on the ascending order of capacity
train_features = train_features[np.argsort(train_features[:, 10])]
#delete the last column of train_features which is the capacity column
train_features = np.delete(train_features, 10, 1)
#train_features = pd.DataFrame(train_features, columns = ['V_Measured', 'I_Measured', 'V / I', 'I_Load',
#    'V / I_Load', 'V/I + V/I_Load', 'Re_Resistance', 'Rct_Resistance', 'Resistance_Added', 'Internal_Temp'])
#print (train_features)
train_labels.sort()
#train_labels = pd.DataFrame(train_labels, columns = ['Capacity'])
#print (train_labels)


#to sort the test_features based on the ascending order of capacity
test_features = test_features[np.argsort(test_features[:, 10])]
#delete the last column of test_features which is the capacity column
test_features = np.delete(test_features, 10, 1)
#test_features = pd.DataFrame(test_features, columns = ['V_Measured', 'I_Measured', 'V / I', 'I_Load',
#    'V / I_Load', 'V/I + V/I_Load', 'Re_Resistance', 'Rct_Resistance', 'Resistance_Added', 'Internal_Temp'])
#print (test_features)
test_labels.sort()
#test_labels = pd.DataFrame(test_labels, columns = ['Capacity'])
#print (test_labels)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
#train_labels = sc.fit_transform(train_labels)
test_features = sc.transform (test_features)
#test_labels = sc.transform (test_labels)

#from sklearn.decomposition import PCA
#pca = PCA(n_components= 6)
#train_features = pca.fit_transform(train_features)
#test_features = pca.transform(test_features)


from sklearn.ensemble import RandomForestRegressor
### Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 250, random_state = 40)
### Train the model on training data
rf.fit(train_features, train_labels);
#from sklearn.metrics import r2_score
#plt.scatter(train_features,train_labels, color = 'red')
#plt.scatter(test_features, test_labels, color = 'green')
#plt.plot(train_features, rf.predict(train_features), color = 'blue')
##plt.plot(xtest, regressor.predict(xtest), color = 'orange')
#plt.title('(Support Vector Regression)')
#plt.xlabel('Re Resistance')
#plt.ylabel('Capacity')
#plt.show()


pred=rf.predict(test_features)

###Accuracy metric
#mse = mean_squared_error(test_features, test_labels)
mse = mean_squared_error(test_labels, pred)
print("Mean Squared Error:",mse)
###Accuracy metric
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(test_labels, pred)
print("Mean Absolute Error:", mae)
