import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.metrics import r2_score

### Read in data and display first 5 rows
features = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\Input_Data\\B5_US.csv")
labels = np.array(features['Capacity'])

### Saving feature names for later use
feature_list = list(features.columns)

### Convert to numpy array
features = np.array(features)

### Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
    test_size = 0.25, random_state = 40)
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

#from sklearn.externals import joblib
#regressor = joblib.load('SVR_model_AllwAuto')

regressor=SVR(kernel='rbf', C=1, gamma=10000, epsilon=0.03)
regressor.fit(train_features,train_labels)
pred=regressor.predict(test_features)

#plt.scatter(train_features, train_labels, color = 'red')
#plt.scatter(test_features, test_labels, color = 'green')
#plt.plot(pred, color = 'blue')
#plt.plot(test_labels, color = 'orange')
#plt.title('(Actual vs Predicted Figures)')
#plt.xlabel('Re Resistance')
#plt.ylabel('Capacity')
#plt.show()

#print(test_labels)
#print('-------------------')
#print(pred)
#data = {'Testing figures': test_labels,
#        'Predicted figures': pred
#        }
#df = pd.DataFrame({'Testing figures': test_labels,
#        'Predicted figures': pred})
#df.to_csv(r'C:\\Users\\Alloy\\Desktop\\FYP_Data\\Output_Data\\B5_Data_SVRw4_auto.csv', index = False, header = True)

###Accuracy metric
#mse = mean_squared_error(test_features, test_labels)
mse = mean_squared_error(test_labels, pred)
print("Mean Squared Error:",mse)
###Accuracy metric
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(test_labels, pred)
print("Mean Absolute Error:", mae)

### ALTERNATIVE WAY TO CALCULATE MAE, Calculate the absolute errors
#errors = abs(pred - test_labels)
### Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
