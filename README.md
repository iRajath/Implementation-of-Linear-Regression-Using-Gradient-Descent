# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary library and load the dataframe.
2. Develop a computeCost function to calculate the cost function.
3. Execute iterations of gradient steps with a specified learning rate.
4. Visualize the cost function using Gradient Descent and create the corresponding graph.

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: S Rajath
RegisterNumber:  212224240127
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    #perform gradient descent
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        #calulate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv')
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)

theta=linear_regression(X1_scaled,Y1_scaled)

new_data=np.array([165349.2,136897.8,471784.1])
new_scaled=scaler.fit_transform(new_data)
new_scaled=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/94abaea8-1254-4595-8887-2da07f96c17f)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
