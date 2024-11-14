# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program & Output :
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: swetha.M
RegisterNumber:  212223040223
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
```

![image](https://github.com/user-attachments/assets/1f531d00-2084-4cf6-94c1-4d598129c3b1)

```
data.tail()
```

![image](https://github.com/user-attachments/assets/96307a61-8efd-48f3-9a3e-736800914289)

```
data.info()
```

![image](https://github.com/user-attachments/assets/daf4a04c-813b-465d-804e-48e3212a8841)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/cc6d99c8-ade2-4058-8fa3-eec2623fbd99)

```
x=data['v2'].values
y=data['v1'].values
y.shape
```

![image](https://github.com/user-attachments/assets/91c469b5-e794-41b1-b31d-9ee33cc9ef35)

```
x.shape
```

![image](https://github.com/user-attachments/assets/9e36511a-f921-4b6a-9dff-d889a756ca65)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```

![image](https://github.com/user-attachments/assets/550de60b-f968-4b1f-974e-3957820c2c9c)

```
y_train.shape
```

![image](https://github.com/user-attachments/assets/b1132566-6a1f-4aa4-9bbe-a60ac1557889)

```
y_test.shape
```

![image](https://github.com/user-attachments/assets/f54626d3-38e9-444c-b365-342d20b5c38f)

```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train.shape
```

![image](https://github.com/user-attachments/assets/a191eda8-6900-44c0-833c-6370697462b6)

```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```

![image](https://github.com/user-attachments/assets/18259235-52df-432f-ab0d-821cd583a566)

```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

![image](https://github.com/user-attachments/assets/82e42158-4063-4d89-8d38-332f9ab40dc8)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
