import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

features = pd.read_csv("/Users\\Alloy\\Desktop\\B5_US.csv")
#print(features.head())

pearsoncorr = features.corr(method='pearson')
#print(pearsoncorr)

sb.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

plt.show()
