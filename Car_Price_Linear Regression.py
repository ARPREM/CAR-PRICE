# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:39:37 2022

@author:PREM
"""
import numpy as np
import pandas as pd
import os
os.chdir("C:/Users/PREM")
data = pd.read_csv("CarPrice_Assignment.csv")
data.shape
data.info()
data.describe()
data.describe(exclude=[object])
data.describe(include=[object])

#categorical column selection
categorical_features = [column_name for column_name in data.columns if data[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)

#Numerical Data Column Selection
numerical_features = [column_name for column_name in data.columns if data[column_name].dtype != 'O']
print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)


#unique values in each columnn
for each_feature in categorical_features:
    unique_values = len(data[each_feature].unique())
    print("Cardinality(no. of unique values) of {} are: {}".format(each_feature, unique_values))

for each_feature in categorical_features:unique_values = len(data[each_feature].unique())
print("Cardinality(no. of unique values) of {} are: {}".format(each_feature, unique_values))

data.head(10)
data.describe(include=[object])

categorical_features = [column_name for column_name in data.columns if data[column_name].dtype == 'O']
data[categorical_features].isnull().sum()


numerical_features = [column_name for column_name in data.columns if data[column_name].dtype != 'O']
data[numerical_features].isnull().sum()


##features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
##for feature in features_with_outliers:
  ####q3 = rain[feature].quantile(0.75)
    ##IQR = q3-q1
    ##lower_limit = q1 - (IQR*1.5)
    ##upper_limit = q3 + (IQR*1.5)
    ##rain.loc[rain[feature]<lower_limit,feature] = lower_limit
    ##rain.loc[rain[feature]>upper_limit,feature] = upper_limit
    
data.isnull().sum
data.fueltype.value_counts()
data.aspiration.value_counts()
data.doornumber.value_counts()
data.carbody.value_counts()
data.drivewheel.value_counts()
#data.enginelocation.value_counts()
data.enginetype.value_counts()
data.cylindernumber.value_counts()
#data.fuelsystem.value_counts()

# encoding categorical data into numberic
data.replace({'fueltype':{'gas':0, 'diesel':1}},inplace=True)
data.fueltype
data.replace({'aspiration':{'std':0, 'turbo':1}},inplace=True)
data.aspiration
data.replace({'doornumber':{'four':4, 'two':2}},inplace=True)
data.replace({'carbody':{'sedan':0, 'hatchback':1,'wagon':2,'hardtop':3,'convertible':4}},inplace=True)
data.replace({'drivewheel':{'fwd':0, 'rwd':1,'4wd':2}},inplace=True)
data.replace({'enginetype':{'ohc':0, 'ohcf':1,'ohcv':2,'l':3,'rotor':4,'dohoc':5}},inplace=True)
data.replace({'cylindernumber':{'four':4, 'six':6, 'five':5, 'eight':8, 'two':2, 'three':3, 'twelve':12}},inplace=True)

data.boxplot('highwaympg')
data.boxplot('citympg')
data.boxplot('horsepower')
data.boxplot('compressionratio')
data.boxplot('stroke')
data.boxplot('cylindernumber')
data.boxplot('carbody')
data.boxplot('enginesize')
data.boxplot('fueltype')
data.boxplot('aspiration')
data.boxplot('doornumber')
data.boxplot('drivewheel')
features_with_outliers = ['highwaympg', 'citympg', 'horsepower', 'compressionratio', 'stroke','cylindernumber', 'carbody', 'enginesize', 'fueltype', 'aspiration', 'doornumber', 'drivewheel']
for feature in features_with_outliers:q1 = data[feature].quantile(0.25) 
q3 = data[feature].quantile(0.75)
IQR = q3-q1
lower_limit = q1 - (IQR*1.5)
upper_limit = q3 + (IQR*1.5)
data.loc[data[feature]<lower_limit,feature] = lower_limit
data.loc[data[feature]>upper_limit,feature] = upper_limit
data.boxplot('highwaympg')
data.boxplot('citympg')
data.boxplot('horsepower')
data.boxplot('compressionratio')
data.boxplot('stroke')
data.boxplot('cylindernumber')
data.boxplot('carbody')
data.boxplot('enginesize')
data.boxplot('fueltype')
data.boxplot('aspiration')
data.boxplot('doornumber')
data.boxplot('drivewheel')

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), linewidths=0.5, annot=False, fmt=".2f", cmap = 'viridis')

data.corr()
y = data[["price"]]
x = data.drop(["price"],axis = 1)
plt.scatter('horsepower','price')
plt.show()
data.info()
newdata = data.drop(["fuelsystem","enginetype","enginelocation","symboling","car_ID","CarName"],axis = 1)
newdata.info()
y = newdata.drop[["price"]]
x = newdata.drop(["price"],axis = 1)
print(x)
x.info()
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

#Model
from sklearn.linear_model import LinearRegression

lin_reg_model = LinearRegression()

lin_reg_model.fit(x_train, y_train)

trainimg_data_prediction = lin_reg_model.predict(x_train)

from sklearn import metrics

error_score = metrics.r2_score(y_train, trainimg_data_prediction)
print("R squared error :", error_score)

#actual vs predicted
plt.scatter(y_train, trainimg_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted")
plt.title("Actual Price vs Predicted Price")
plt.show()

testing_data_prediction = lin_reg_model.predict(x_test)
error_score = metrics.r2_score(y_test, testing_data_prediction)
print("R squared error :", error_score)

plt.scatter(y_test, testing_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted")
plt.title("Actual Price vs Predicted Price")
plt.show()
print("R squared error :", error_score)



lin_reg_model.fit(x_test, y_test)

test__prediction = lin_reg_model.predict(x_test)

from sklearn import metrics

error_score1 = metrics.r2_score(y_test, test__prediction)
print("R squared error :", error_score1)

#actual vs predicted
plt.scatter(y_test, test__prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted")
plt.title("Actual Price vs Predicted Price")
plt.show()
