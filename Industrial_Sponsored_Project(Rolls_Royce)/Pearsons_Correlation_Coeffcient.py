##### This kernel will showcase how to import a dataset and use Pearson's correlation coefficient on it
#If you wish to use a normal correlation coefficient instead, you may just remove the method = 'pearson' while fitting

# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Import the dataset
features = pd.read_csv("dataset.csv")
#print(features.head())

# Setting up the Pearson's correlation coefficient
pearsoncorr = features.corr(method='pearson')

# Using seaborn to plot out the heatmap
# Using a heatmap will allow us to visualise  and narrow down on what we intend to do which is to find higher correleation targets more easily
sb.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

plt.show()
