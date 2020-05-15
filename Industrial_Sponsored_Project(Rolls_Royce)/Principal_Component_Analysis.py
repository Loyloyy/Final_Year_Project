# This kernel will showcase my work while learning and building up the script for creating PCA separately
# This script can also be used to

# Import the relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Import the dataset
df = pd.read_csv("C:/Users/Alloy/Desktop/Final_Year_Project/FYP_ML_Scripts/FYP_Data/Input_Data/B5_US.csv")

# Scale the dataset accordingly
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# Import the PCA library
from sklearn.decomposition import PCA

# Fit the PCA with the desired number accordingly
pca = PCA(n_components=4)

# Fit the algorithm to the scaled data
pca.fit(scaled_data)

# Transform the data
x_pca = pca.transform(scaled_data)


# From here, we can utilise 3 methods to see how effective PCA is
# Firstly, we can create a variance ratio to determine how much varirance is in each of the newly created PCA test_features
# The reason for this is because we can then further elimate any further components/features if the variance is very low (usually below 0.X%)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# Another method involves plotting the PCA components onto a heatmap. This is basically the same as the variance ratio above but in heatmap form for visualisation
map= pd.DataFrame(pca.components_,columns=['V_Measured', 'I_Measured', 'V / I', 'I_Load', 'V / I_Load',
      'V/I + V/I_Load', 'Re_Resistance', 'Rct_Resistance', 'Resistance_Added',
      'Internal_Temp', 'Capacity'])
plt.figure(figsize=(12,6))
sns.heatmap(map,cmap='RdBu_r')
plt.show()

# The last method method we can utilise it to plot out the data on a graph
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],cmap='rainbow')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
