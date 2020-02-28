import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
# Tinh cho linear goc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
#Tinh theo ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
#Tinh theo lasso
from sklearn.linear_model import Lasso
from sklearn import linear_model
import math

##Plot function
def scatter_plot(data, feature, target):
	plt.figure (figsize =(16,8))
	plt.scatter (
		data[feature],
		data[target],
		c= 'black')
	plt.xlabel ("Time")
	plt.ylabel ("Price")
	

def plot_2_figures(df):
	columns = df.columns
	plt.plot(df.index, df[columns[0]] / 10, label = 'DJI', color = 'red')
	plt.plot(df.index, df[columns[1]], label = 'VNIndex', color = 'blue')
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.title('Compare 2 stock')
	plt.legend()
	plt.show()

#Create empty dataframe with VNIndex and DJ
df = pd.DataFrame(columns = ['Date', 'DJI', 'VNI'])
#Read DJI
df1 = pd.read_csv('^DJI.csv')
#Read VNIndex
df2 = pd.read_csv('VNINDEX.csv')
#Take the Date columns from VNIndex dataframe, because VNIndex has less records than DJ
df.Date = df2.DATE
#Change date into index
df.set_index('Date', inplace = True) 
df1.set_index('Date', inplace = True)
df2.set_index('DATE', inplace = True)
#Change string format of date columns into datetime
df.index = pd.to_datetime(df.index)
df1.index = pd.to_datetime(df1.index)
df2.index = pd.to_datetime(df2.index)

#Merge both Price into one dataframe
for each_day in list(df.index):
	if each_day in set(df1.index) and each_day in set(df2.index):
		df['DJI'][each_day] = df1['Close'][each_day]
		df['VNI'][each_day] = df2['CLOSE'][each_day]

#Test print the data
print(df)
plot_2_figures(df)

#Remove nan value
#remove is an array of index that contains nan value
remove = df[pd.isnull(df['DJI'])][pd.isnull(df['VNI'])].index
#Replace nan with previous value
for i in range(len(df)):
	if math.isnan(df['DJI'][i]):
		df['DJI'][i] = df['DJI'][i - 1]
	if math.isnan(df['VNI'][i]):
		df['VNI'][i] = df['VNI'][i - 1]

#Now run regression
regr = linear_model.LinearRegression()
regr.fit(np.array(df.DJI).reshape(-1, 1), np.array(df.VNI).reshape(-1, 1))
print('Coefficient bewteen the 2 is: ', regr.coef_[0])


