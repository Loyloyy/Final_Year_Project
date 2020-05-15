##### This kernel will showcase how to import a dataset and use the Random Forest algorithm (Regression) on it

### Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


### Import the dataset
features = pd.read_csv("dataset.csv")

# Separate the label (values we want to predict) from the dataset
labels = np.array(features['Capacity'])

# Convert the dataset to numpy array
# DO NOT DROP THE LABEL COLUMN YET
features = np.array(features)


### Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

### Sort the rows based on the labels column (Capacity)
# You may find out more about why I decided to sort the columns in the README section
# Firstly, sort the entire dataset based on the labels column
train_features = train_features[np.argsort(train_features[:, 10])]

# Then delete, the capacity column
train_features = np.delete(train_features, 10, 1)

# As the labels column has already been separate, I shall also sort it here separately
##### Update: I realised that I could have just extracted the sorted labels column to improve the time complexity of this script
train_labels.sort()

# Repeat the sort steps above for the testing set
test_features = test_features[np.argsort(test_features[:, 10])]
test_features = np.delete(test_features, 10, 1)
test_labels.sort()

### As I was facing a regression problem, the figures had a huge difference in their scale
# Hence, scaling columns with a wide range of values was very important
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform (test_features)

### Using Principal Component Analysis to reduce the dimensions of the data
# When not in use/testing, just change this section to remarks
from sklearn.decomposition import PCA
pca = PCA(n_components= 6)
# IMPORTANT to only use fit.transform once here, then just transform the test set
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)


### Import the Random Forest Regressor library
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor object and tune it accordingly
rf = RandomForestRegressor(n_estimators = 250, random_state = 40)

# Then, fit the Regressor with the training data to create a model
rf.fit(train_features, train_labels)

# Use the model to predict based on the test set
pred=rf.predict(test_features)


### Lastly, check for the accuracy of the test set
# Checking the Mean Squared Error (MSE)
mse = mean_squared_error(test_labels, pred)
print("Mean Squared Error:",mse)

# Checking the Root Mean Squared Error (RMSE) by simply rooting the MSE
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# # Checking the Mean Aboslute Error(MAE)
mae = mean_absolute_error(test_labels, pred)
print("Mean Absolute Error:", mae)
