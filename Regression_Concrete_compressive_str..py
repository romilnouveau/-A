# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:08:05 2020

@author: shreya
"""

import pandas as pd
cc_data=pd.read_csv(r'C:\Users\shreya\Desktop\Concrete_Data.csv')

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

lm=cc_data.drop('CCS(MPa)',axis=1)
x = lm.to_numpy()
y = np.array(cc_data['CCS(MPa)'])
y=y[:,np.newaxis]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
rmse=np.sqrt(mean_squared_error(y_test,y_predict))
r2=r2_score(y_test,y_predict)
print(rmse)
print(r2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)

x_poly_train = polynomial_features.fit_transform(x_train)
x_poly_test = polynomial_features.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_poly_train,y_train)
y_predict=model.predict(x_poly_test)


from sklearn.metrics import mean_squared_error, r2_score
rmse=np.sqrt(mean_squared_error(y_test,y_predict))
r2=r2_score(y_test,y_predict)
print(rmse)
print(r2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=3)

x_poly_train = polynomial_features.fit_transform(x_train)
x_poly_test = polynomial_features.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_poly_train,y_train)
y_predict=model.predict(x_poly_test)


from sklearn.metrics import mean_squared_error, r2_score
rmse=np.sqrt(mean_squared_error(y_test,y_predict))
r2=r2_score(y_test,y_predict)
print(rmse)
print(r2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from sklearn import neighbors
model=neighbors.KNeighborsRegressor(2)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=200, test_size=0.2)
cv_r2 = cross_val_score( model, x, y, scoring='r2', cv=cv)
#print(cv_r2)                            
import numpy as np
print(np.mean(cv_r2))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from sklearn import neighbors
model=neighbors.KNeighborsRegressor(3)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=200, test_size=0.2)
cv_r2 = cross_val_score( model, x, y, scoring='r2', cv=cv)
#print(cv_r2)                            
import numpy as np
print(np.mean(cv_r2))