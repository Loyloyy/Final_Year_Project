##### This kernel will showcase how to import a dataset and use the Extreme Learning Machine algorithm (Regression) on it

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


### Import the Extreme Learning Machine Regressor library
from elm import ELMRegressor

# Create a Extreme Learning Machine Regressor object and tune it accordingly
# In this case, I decided to create more hidden layers as it was consistently performing much better than a single layer without too much of a sacrifice to time performance
elm = ELMRegressor(n_hidden=30)

# Then, fit the Regressor with the training data to create a model
elm.fit(train_features, train_labels)

# The next section is different from the other algorithms as ELM's hidden layer constantly changes their parameters
# Hence, the following for loop will allow us to iterate through the ELM model multiple times to find the mean
total = 0
rmse_val = []                                           # To store rmse values for different x
for x in range(20):
    x = x+1
    model = ELMRegressor(n_hidden=30)
    model.fit(train_features, train_labels)             # Fit the model
    pred=model.predict(test_features)                   # Make prediction on test set
    error = sqrt(mean_squared_error(test_labels,pred))  # Calculate rmse
    total = total + error
    rmse_val.append(error)                              # Store rmse values
    print('RMSE value for x = ' , x , 'is:', error)
avg = total / x
print('Average RMSE value for x =', avg, )


# The foolowing for loop will allow us to constant repeat the ELM modelling to find the best accuracy
# Once the best accuracy is found from the loop, it can then be saved using the joblib function under sklearn
# More about the joblib function is explained in README
temp = 1
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = ELMRegressor(n_hidden=30)
    model.fit(train_features, train_labels)  #fit the model
    pred=model.predict(test_features) #make prediction on test set
    error = sqrt(mean_squared_error(test_labels,pred)) #calculate rmse
    if (error < temp ):
        temp = error
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
#avg = total / K
print('Best RMSE value for k = ', temp)
