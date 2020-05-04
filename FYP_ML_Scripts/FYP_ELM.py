import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

### Read in data and display first 5 rows
features = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\Input_Data\\B6b.csv")
labels = np.array(features['Capacity'])

### Saving feature names for later use
feature_list = list(features.columns)

### Convert to numpy array
features = np.array(features)


### Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#to sort the train_features based on the ascending order of capacity
train_features = train_features[np.argsort(train_features[:, 4])]
#delete the last column of train_features which is the capacity column
train_features = np.delete(train_features, 4, 1)
train_labels.sort()
#to sort the test_features based on the ascending order of capacity
test_features = test_features[np.argsort(test_features[:, 4])]
#delete the last column of test_features which is the capacity column
test_features = np.delete(test_features, 4, 1)
test_labels.sort()

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


from sklearn.externals import joblib
elm = joblib.load('ELM_model_Selected')


from elm import ELMRegressor
#elm = ELMRegressor()
elm.fit(train_features, train_labels)
#train_res.append(e.score(train_features, train_labels))
#test_res.append(e.score(test_features, test_labels))


pred=elm.predict(test_features)

#print(test_labels)
#print('-------------------')
#print(pred)

#from sklearn.externals import joblib
#joblib.dump(elm, 'ELM_model_SelectedwAuto')

#accuracy matrix
mse = mean_squared_error(test_labels, pred)
print("Mean Squared Error:",mse)
###Accuracy metric
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(test_labels, pred)
print("Mean Absolute Error:", mae)
