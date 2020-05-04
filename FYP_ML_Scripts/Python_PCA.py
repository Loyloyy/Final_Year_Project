import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#%matplotlib inline

df = pd.read_csv("/Users\\Alloy\\Desktop\\FYP_Data\\Input_Data\\B56718b.csv")
#print(df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

#print(scaled_data.shape)
#print(x_pca.shape)


#plt.figure(figsize=(8,6))
#plt.scatter(x_pca[:,0],x_pca[:,1],c=pd['target'],cmap='rainbow')
#plt.xlabel('First principal component')
#plt.ylabel('Second Principal Component')


pca.components_
#map= pd.DataFrame(pca.components_,columns=['V_Measured', 'I_Measured', 'V / I', 'I_Load', 'V / I_Load',
#       'V/I + V/I_Load', 'Re_Resistance', 'Rct_Resistance', 'Resistance_Added',
#       'Internal_Temp', 'Capacity'])
#plt.figure(figsize=(12,6))
#sns.heatmap(map,cmap='RdBu_r')
#plt.show()
