# Data Science Clustering
## Contents
-  [Overview](#over-view)
-  [Data Preparation](#data-preparation)
-  [Data Classification](#data-classification)
    - [Decision Tree Without Key Parameter](#without-key-parameter)
    - [Decision Tree With Appropriate Key Prameter](#with-key-parameter)
- [Data Clustering](#data-clustering)
## Overview

This project includes a method for classifying data and highlights the importance of key parameters in the decision tree. Additionally, the number of clusters is determined based on the elbow chart. The README file provides specific explanations for key lines of code and charts. This project is expected to provide basic knoweldge of the data classification and clustering 

## 1 Data preparation
Through the Data preparation part, the given data was cleaned. A critical problem was some colums include a non-numeric values
To clear those problems 
### Change to numeric values
```
data['name of the colum'] = pd.to_numeric(data['name of the colum'], errors='coerce')
```
This line of colum halps to change the data types of the certain colume to a numeric value

After cleanning the data, a sample random dataset was taken. To create sample random dataset, the line of code below was used.

```
random_indices = random.sample(range(num_rows), 600)
random_sample = data.iloc[random_indices]
random_sample = random_sample.dropna()
```
The distribution of the sample random dataset is shown through the charts below.

<img width="797" alt="image" src="https://github.com/hyeonBin7201/Data-science-clustering/assets/152157238/84ea0b60-7e79-42b8-8303-616d791b5680">


## 2 Data Classification (Decision Tree)

Task 2 aims to classify the dataset. Based on the columns, excluding the 'quality' column, the decision tree will predict the quality level for each data point. After the prediction, the predicted quality levels are compared with the actual quality levels. Finally, the accuracy of the predicted quality levels is presented through metrics.

During the creation of the decision tree, the trees were generated several times. Firstly, the decision tree was created without any justification for key parameters. Subsequently, the decision tree was refined gradually with appropriate key parameter values.

### Without key parameter
The way to generate the decision tree is below
```
X_train, X_test, Y_train, Y_test = train_test_split(input, target,train_size = 0.7, random_state = 14)
clf = DecisionTreeClassifier()
fit = clf.fit(X_train, Y_train)

fig = plt.figure(figsize=(10,10))
fig = plot_tree(clf,
              feature_names=input.columns.values,
              class_names=list(map(str, target.unique())),
              rounded=True,
              filled=True)
plt.show()
```

<img width="573" alt="image" src="https://github.com/hyeonBin7201/Data-science-clustering/assets/152157238/7e08dade-07a3-4748-8fc8-057643568f7b">

Moreover, to create metric to check the decision tree's accuracy
```
from sklearn.metrics import classification_report
Y_pre = fit.predict(X_test)
print(classification_report(Y_test,Y_pre))
```

<img width="475" alt="image" src="https://github.com/hyeonBin7201/Data-science-clustering/assets/152157238/be64d5b2-4074-40b2-aaeb-e6bb283e6071">

### with appropriate key parameter

The key parameters which were checked whether it is affected to the accurarcy of the prediction result were max depth,max leaf nodes,ccp alpha and max features.
justifying the key prameter hleps avoid the overfitting and imporve the prediction accuracy.
<img width="566" alt="image" src="https://github.com/hyeonBin7201/Data-science-clustering/assets/152157238/84fe24c6-efa0-4838-a7e5-bc0ee7f40ded">

## 3 Data Clustering
### clustering elbow chart
With the final decision tree, the data were clustered. During the data clustering process, the number of clusters plays an important role. To determine the appropriate number of clusters, the elbow chart was used. In this case, the number of clusters was decided to be 9.
<img width="627" alt="image" src="https://github.com/hyeonBin7201/Data-science-clustering/assets/152157238/725b8e37-e1e9-480e-98eb-8d4955e955b3">

The way to create the elbow chart is below
```
for i in range(1,100):
    model = KMeans(n_clusters = i,init='k-means++',n_init = 20, random_state=43)
    fit = model.fit(Copy_Data_x_scaled)
    distortions.append(model.inertia_)
    distortion = model.inertia_

    if target_distortion <= distortion:
        selected_clusters = i

plt.plot(range(1,100),distortions, marker='o')
plt.tight_layout()
plt.show

print(f"When distortion is Distortion >= 3000, the number of cluster : {selected_clusters}")
```
This code create the elbow chart with the number of cluster from 1 to 100
