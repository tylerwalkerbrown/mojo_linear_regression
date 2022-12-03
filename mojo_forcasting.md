# Mojo Analysis


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline
# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
# For time stamps
from datetime import datetime
import datetime as dt
import os
```


```python
#For deep learning = spray_by_date
"""WITH seven_figures as (
select  *
 ,substring(completeddate,1,2) as month1, 
substring(completeddate,3,2) as Days1,
substring(completeddate,5,3) as year1,
length(completeddate) as len
from sprays_by_date)
, a as (
select *,trim('/' from month1) as month ,
trim('/' from year1) as year,
trim('/' from Days1) as day
from seven_figures
)
,b as (
select *,concat(20,year,'/', month, '/', day) as complete
from a
)
,c as (
select *,cast(complete as datetime) as date
from b
)
select branchname,routename,TIME, accountnum, description,
programname,woheaderid, Businessname as customer_name,
fullAddress,streetnumber, STATE, city, postalcode, SumOfbillamount,
employee, propertytype, duration, time_in, timeout,date,targetpest
from c
"""
```




    "WITH seven_figures as (\nselect  *\n ,substring(completeddate,1,2) as month1, \nsubstring(completeddate,3,2) as Days1,\nsubstring(completeddate,5,3) as year1,\nlength(completeddate) as len\nfrom sprays_by_date)\n, a as (\nselect *,trim('/' from month1) as month ,\ntrim('/' from year1) as year,\ntrim('/' from Days1) as day\nfrom seven_figures\n)\n,b as (\nselect *,concat(20,year,'/', month, '/', day) as complete\nfrom a\n)\n,c as (\nselect *,cast(complete as datetime) as date\nfrom b\n)\nselect branchname,routename,TIME, accountnum, description,\nprogramname,woheaderid, Businessname as customer_name,\nfullAddress,streetnumber, STATE, city, postalcode, SumOfbillamount,\nemployee, propertytype, duration, time_in, timeout,date,targetpest\nfrom c\n"



# Data Cleansing and Reformatting


```python
#Changing Directory 
os.chdir('Desktop/mojo/fresh')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Input In [41], in <cell line: 2>()
          1 #Changing Directory 
    ----> 2 os.chdir('Desktop/mojo/fresh')


    FileNotFoundError: [Errno 2] No such file or directory: 'Desktop/mojo/fresh'



```python
#Reading in data 
spray_by_date = pd.read_csv('spray_by_dates.csv')
#Customer info for billamount
info = pd.read_csv('customer_information.csv')
```


```python
#spray_by_date.replace(0, np.nan, inplace=True)
#spray_by_date.replace(np.nan,-93, inplace=True)
```


```python
#Setting index to the date 
spray_series = spray_by_date.set_index(spray_by_date.date)
```


```python
# Replacing 0s (resrpays) with negative average spray amount 
spray_by_date['SumOfbillamount'] = spray_by_date.replace(to_replace = 0, 
                                      value = -int(spray_by_date.SumOfbillamount.mean()), inplace=True)
```


```python
#Creating a datetime coulmn storing as a datetime 
spray_series['datetime'] = pd.to_datetime(spray_series.date)
```


```python
#Day of the year 
spray_series['dayofyear'] = spray_series.datetime.dt.dayofyear.astype(int)
```


```python
#Filtering out the data by year 
spray_series_2020 = spray_series.loc[(spray_series['datetime'] > '2020-01-05') 
                                     & (spray_series['datetime'] < '2021-01-05')]
spray_series_2021 = spray_series.loc[(spray_series['datetime'] > '2021-01-05') 
                                     & (spray_series['datetime'] < '2022-01-05')]
spray_series_2022 = spray_series.loc[(spray_series['datetime'] > '2022-01-05') 
                                     & (spray_series['datetime'] < '2023-01-05')]
```


```python
#finding the day of year to plot 
spray_series_2022['dayofyear'] = spray_series_2022.datetime.dt.dayofyear
spray_series_2021['dayofyear'] = spray_series_2021.datetime.dt.dayofyear
spray_series_2020['dayofyear'] = spray_series_2020.datetime.dt.dayofyear
```

    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_7729/1213484696.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      spray_series_2022['dayofyear'] = spray_series_2022.datetime.dt.dayofyear
    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_7729/1213484696.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      spray_series_2021['dayofyear'] = spray_series_2021.datetime.dt.dayofyear
    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_7729/1213484696.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      spray_series_2020['dayofyear'] = spray_series_2020.datetime.dt.dayofyear



```python
amount_2020 =  pd.DataFrame(spray_series_2020.SumOfbillamount.values.astype(int),index = spray_series_2020.datetime.dt.dayofyear, columns = ['billamount'])
amount_2021 =  pd.DataFrame(spray_series_2021.SumOfbillamount.values.astype(int),index = spray_series_2021.datetime.dt.dayofyear, columns = ['billamount'])
amount_2022 = pd.DataFrame(spray_series_2022.SumOfbillamount.values.astype(int),index = spray_series_2022.datetime.dt.dayofyear, columns = ['billamount'])
```


```python
#Creating a cumulative summary of revenue 
spray_series['cumulative_revenue'] = spray_series.SumOfbillamount.cumsum()
```


```python
#Data types in frame
spray_series.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4277 entries, 2020-06-05 00:00:00 to 2022-10-06 00:00:00
    Data columns (total 25 columns):
     #   Column              Non-Null Count  Dtype         
    ---  ------              --------------  -----         
     0   branchname          4277 non-null   object        
     1   routename           4277 non-null   object        
     2   TIME                4277 non-null   object        
     3   accountnum          4277 non-null   int64         
     4   description         4277 non-null   object        
     5   programname         4277 non-null   object        
     6   woheaderid          4277 non-null   int64         
     7   customer_name       4277 non-null   object        
     8   fullAddress         4277 non-null   object        
     9   streetnumber        4178 non-null   object        
     10  STATE               4277 non-null   object        
     11  city                4277 non-null   object        
     12  postalcode          4277 non-null   int64         
     13  SumOfbillamount     4277 non-null   int64         
     14  employee            4277 non-null   object        
     15  propertytype        4277 non-null   object        
     16  duration            4277 non-null   int64         
     17  time_in             4277 non-null   object        
     18  timeout             4277 non-null   object        
     19  date                4277 non-null   object        
     20  targetpest          4107 non-null   object        
     21  month_day           4277 non-null   object        
     22  datetime            4277 non-null   datetime64[ns]
     23  dayofyear           4277 non-null   int64         
     24  cumulative_revenue  4277 non-null   int64         
    dtypes: datetime64[ns](1), int64(7), object(17)
    memory usage: 868.8+ KB


# EDA


```python
#Packages
import seaborn as sns
```


```python
#Summary statisistics for all columns
spray_series.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accountnum</th>
      <th>woheaderid</th>
      <th>postalcode</th>
      <th>SumOfbillamount</th>
      <th>duration</th>
      <th>dayofyear</th>
      <th>cumulative_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.277000e+03</td>
      <td>4.277000e+03</td>
      <td>4277.000000</td>
      <td>4277.000000</td>
      <td>4277.000000</td>
      <td>4277.000000</td>
      <td>4277.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.963321e+05</td>
      <td>1.383775e+07</td>
      <td>4393.119009</td>
      <td>99.173252</td>
      <td>21.017769</td>
      <td>189.737667</td>
      <td>207398.594342</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.997790e+04</td>
      <td>1.855830e+06</td>
      <td>1100.282074</td>
      <td>39.208882</td>
      <td>26.519212</td>
      <td>42.155032</td>
      <td>123043.699456</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.708580e+05</td>
      <td>9.014951e+06</td>
      <td>3046.000000</td>
      <td>0.000000</td>
      <td>-693.000000</td>
      <td>94.000000</td>
      <td>67.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.657000e+05</td>
      <td>1.219087e+07</td>
      <td>3281.000000</td>
      <td>85.000000</td>
      <td>10.000000</td>
      <td>158.000000</td>
      <td>102103.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.849190e+05</td>
      <td>1.485212e+07</td>
      <td>3846.000000</td>
      <td>93.000000</td>
      <td>18.000000</td>
      <td>191.000000</td>
      <td>202833.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.046353e+06</td>
      <td>1.529700e+07</td>
      <td>5456.000000</td>
      <td>105.000000</td>
      <td>26.000000</td>
      <td>223.000000</td>
      <td>314739.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.096973e+06</td>
      <td>1.612165e+07</td>
      <td>13030.000000</td>
      <td>606.000000</td>
      <td>574.000000</td>
      <td>279.000000</td>
      <td>424164.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Changing the box plot size and layout
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.autolayout"] = True
```


```python
#Comparing the results of the predicted vs the actual
amount_2020.billamount.cumsum().plot(linewidth=2.0)
amount_2021.billamount.cumsum().plot(linewidth=2.0)
amount_2022.billamount.cumsum().plot(linewidth=2.0)
plt.xlabel('Days', fontsize=15)
plt.ylabel('Revenue', fontsize=15)
plt.xticks(ha='right', rotation=55, fontsize=15, fontname='monospace')
plt.yticks(rotation=55, fontsize=15, fontname='monospace')
plt.title('Cumulative Summary of Revenue', fontsize=15)
plt.legend(loc=2,prop={'size': 10})
plt.show()
```


    
![png](output_20_0.png)
    



```python
#Graphing the day of year and summary of bill amount per day
amount_2020.billamount.plot(linewidth=2.0, color = 'b',alpha = .3, label = '2020')
amount_2021.billamount.plot(linewidth=2.0, color = 'g',alpha = .3, label = '2021')
amount_2022.billamount.plot(linewidth=2.0, color = 'r',alpha = .3, label = '2022')
plt.xlabel('Amount Made Per Day', fontsize=15)
plt.ylabel('Revenue', fontsize=15)
plt.xticks(ha='right', rotation=55, fontsize=15, fontname='monospace')
plt.yticks(rotation=55, fontsize=15, fontname='monospace')
plt.title('Cumulative Summary of Revenue', fontsize=15)
plt.legend(loc=2,prop={'size': 10})
plt.show()
```


    
![png](output_21_0.png)
    



```python
#Rolling mean of the average sum amount per week
amount_2020.groupby(['datetime']).sum().billamount.rolling(7).mean().plot(linewidth=2.0, color = 'b', label = '2020')
amount_2021.groupby(['datetime']).sum().billamount.rolling(7).mean().plot(linewidth=2.0, color = 'g', label = '2021')
amount_2022.groupby(['datetime']).sum().billamount.rolling(7).mean().plot(linewidth=2.0, color = 'r', label = '2022')
plt.xlabel('Amount Made Per Day', fontsize=15)
plt.ylabel('Amount made', fontsize=15)
plt.xticks(ha='right', rotation=55, fontsize=15, fontname='monospace')
plt.yticks(rotation=55, fontsize=15, fontname='monospace')
plt.title('Rolling Mean of Bill Amount', fontsize=15)
plt.legend(loc=2,prop={'size': 10})
plt.show()
```


    
![png](output_22_0.png)
    



```python
#Correlation matrix 
#Correlation Heat Map 
matrix = spray_series.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()
```


    
![png](output_23_0.png)
    


Suspected to see correlation between day of year and bill amount

# Linear Regression


```python
from sklearn.linear_model import LinearRegression
```


```python
avg_sales = pd.DataFrame(spray_series.SumOfbillamount)
# Create a time dummy
time = np.arange(len(spray_series.SumOfbillamount))

avg_sales['time'] = time 

# Create training data
X = avg_sales.loc[:, ['time']]
y = avg_sales.loc[:, 'SumOfbillamount']

# Train the model
model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
```


```python
ax = y.plot(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Average Sales trend over years');
```


    
![png](output_28_0.png)
    



```python
avg_sales2 = pd.DataFrame(spray_series.groupby(spray_series.index)['SumOfbillamount'].sum())
# Create a time dummy
time2 = np.arange(len(spray_series.groupby(spray_series.index)['SumOfbillamount'].sum()))

avg_sales2['time'] = time2 

# Create training data
X2 = avg_sales2.loc[:, ['time']]
y2 = avg_sales2.loc[:, 'SumOfbillamount']

# Train the model
model2 = LinearRegression()
model2.fit(X2, y2)

y_pred2 = pd.Series(model2.predict(X2), index=X2.index)
```


```python
ax2 = y2.plot(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False, alpha=0.5)

ax2 = y_pred2.plot(ax=ax2, linewidth=3)
ax2.set_title('Daily Earnings ');
```


    
![png](output_30_0.png)
    

