# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values from dataframe and apply label encoder.

3.Apply decision tree classifier on the dataframe.

4.obtain the value of accuracy and data prediction.

## Program:
```
Developed by: Safeeq Fazil.A
RegisterNumber:  212222240086

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

### Initial Dataset:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/72142d0f-2542-4dd5-8b21-5a5f75341a1b)

### data info:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/79316151-db55-43d0-aad9-b0a92646a7e6)

### null values:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/3590c562-4223-46a6-b3a3-631a8d641852)

### assignment of x and y values:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/80e1aadf-4668-4c29-9d4f-7e5cbf184420)

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/24c9f6de-5845-49da-b9c3-880d6a528e5b)

### Converting string literals to numerical values using label encoder:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/37280f6d-4485-4949-9955-3ec3b5eea0d8)

### Accuracy:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/ddf0360f-5660-48ad-b3c4-8614fd1c093e)

### Prediction:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680361/04e0f7e1-2552-46a2-8d49-110a2648d449)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
