import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# import dataset
features = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\Input_Data\\B6_US.csv")

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
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
train_features = train_features[np.argsort(train_features[:, 10])]
train_features = np.delete(train_features, 10, 1)
train_labels.sort()
test_features = test_features[np.argsort(test_features[:, 10])]
test_features = np.delete(test_features, 10, 1)
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


# import the regressor
from sklearn.tree import DecisionTreeRegressor

# create a regressor object
DTR = DecisionTreeRegressor(max_depth = 15, random_state = 40)

# fit the regressor with X and Y data
DTR.fit(train_features, train_labels)
pred= DTR.predict(test_features)

mse = mean_squared_error(test_labels, pred)
print("Mean Squared Error:",mse)
###Accuracy metric
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(test_labels, pred)
print("Mean Absolute Error:", mae)

# arange for creating a range of values
# from min value of X to max value of X
# with a difference of 0.01 between two
# consecutive values
#X_grid = np.arange(min(train_features), max(train_features), 0.01)

# reshape for reshaping the data into
# a len(X_grid)*1 array, i.e. to make
# a column out of the X_grid values
#X_grid = X_grid.reshape((len(X_grid), 1))

# scatter plot for original data
#plt.scatter(train_features, train_labels, color = 'red')

# plot predicted data
#plt.plot(X_grid, DTR.predict(X_grid), color = 'blue')
#plt.title('Profit to Production Cost (Decision Tree Regression)')
#plt.xlabel('Production Cost')
#plt.ylabel('Profit')
#plt.show()


# import export_graphviz
#from sklearn.tree import export_graphviz

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
#export_graphviz(DTR, out_file ='tree.dot',
#               feature_names =['Production Cost'])
