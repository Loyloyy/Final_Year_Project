# Industrial Sponsored Project with Rolls Royce (aka FYP)
## Title: Machine Learning on Battery Performance and Health Management

### Synopsis
The scripts and kernels included in this repository showcases my work that I have done and created through my industrial sponsored project. The aim of the project was to find ways to do feature selection for the degradation of battery's health. As such, I decided to utilise methods such as finding the correlation coefficient, principal component analysis and data wrangling to create new features that allow us to better predict the battery's capacity through the cycles

The results are not mentioned here due to confidentiality as it is a industrial sponsored project.

### Content and algorithms I used
With regards to data extraction and automation, the primary languages I utilised included C# and MATLAB.

For feature engineering, the main algorithms and methods I used includes
- Pearson's Correlation Coefficient
- Principal Component Analysis
- Data Wrangling

Algorthims I used included,
- Support Vector Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbour
- Extremel Learning Machine


### Code Explanations
I will like to highlist and reiterate some important things to note if anyone wants to understand several of the actions I took.

Firstly, it is very important to scale the data before using PCA. More information can be found in the link here, https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca


Also, one confusing step might be to sort the rows of values based on the descending order of the labels column (Capacity). The main motivation was because in not doing so, the model created will not be a smooth regression line predicting the degradation of capacity. Instead, it will constantly go from a top value to the middle/bottom and up again constantly, creating a messy sphagetti look as shown below.

![Alt Text](https://user-images.githubusercontent.com/64775878/82097674-f2282700-9735-11ea-92dd-b7d8d7e66d08.jpg)

I eventually determined the cause was due to the fluctuating capacity of the battery test. For example, we will expect the capacity to degrade by X% after several cycles, however in some instances, the capacity went up instead. I did not pursue the reasoning of the fluctuating capacity as it was outside the scope of my ISP but classified it as a potential future work to be conducted

Without sorting the rows, the values predicted had a awful RMSE value. The RMSE improved frastically after sorting the rows as well.


Another issue to take note of will be to use fit.transform and transform accordingly and not just one 'fit.transform' will be to craete a new settings for the data you wish to use it on while 'transform' will infer from the row you use 'fit.transform' and apply the same settings. Using 'fit.transform' continously will result in bad accuracy.


With regards to multiple testing of several models such as Extreme Learning Machine, the best model can be extracted using the joblib function from sklearn.externals by using joblib.dump. The same function can then be used to load the saved mode by using joblib.load to load the model into the new script.



Thank you for reading and if you have any suggestions on how I can improve, feel free to reach out to me at atan091@e.ntu.edu.sg
