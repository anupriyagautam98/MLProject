import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
data1 = pd.read_csv('CO2 emission old.csv')
data2 = data1[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#print(data2.head(10))
data1.shape

print(data2.describe())

print(data2.dtypes)
#fig = plt.figure()   #for allocating space
data2.hist()         #for drawing histogram for each column
plt.tight_layout
plt.show()           #for printing the graph
cor = data2.corr()      #for printing table for correlation of each row with every col
print(cor)

sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns) #graphical representaion for corr
plt.show()

fig = plt.figure(figsize=(50,30))
a1 = fig.add_subplot(221)       #2by2 matrix and 1st position
a2 = fig.add_subplot(222)
a3 = fig.add_subplot(223)
sns.scatterplot(x='ENGINESIZE',y='CO2EMISSIONS', data=data2,ax=a1)
sns.scatterplot(x='CYLINDERS',y='CO2EMISSIONS', data=data2,ax=a2)
sns.scatterplot(x='FUELCONSUMPTION_COMB',y='CO2EMISSIONS', data=data2,ax=a3)
plt.show()

  #from the graph we opt to take fuelconsumption as independent variable

x = data2[['FUELCONSUMPTION_COMB']]
y = data2[['CO2EMISSIONS']]
trainx,testx,trainy,testy = train_test_split(x,y, test_size=0.2, random_state=20)
                                         #splitting the data into train and test data sets
model = linear_model.LinearRegression()  #creating linear regression object...to create an empty set
model.fit(trainx, trainy.values.ravel())    #to train the  model
result= model.score(trainx, trainy)
print('accuracy:',result)
print('slope: ',model.coef_)
print('intercept:',model.intercept_)
  #testing...
predictY = model.predict(testx)    #for predicting y by giving the values of x
print(predictY)
print(predictY.shape)
print('error:',metrics.mean_squared_error(testy,predictY))    #comparing testy with predictY
plt.scatter(testx,testy, color='r', label='actual data')
plt.plot(testx,predictY, color='c', label='predicted data')
plt.xlabel("FUELCONSUMPTION")
plt.ylabel("CO2EMISSIONS")
plt.legend()
plt.show()
