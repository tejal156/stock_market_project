#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install yfinance


# installing yfinance

# In[28]:


import yfinance as yf

# msft = yf.Ticker("AAPL")


# # downloding the dataset

# In[29]:


df1 = yf.download("msft")
df2 = yf.download("AAPL")
df3 = yf.download("GOOG")
df4 = yf.download("SPY")


# # functions for modifying dataset

# In[30]:


# this function add important column to data which are RSI
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi


# In[31]:


# this function add important column to data which is MACD
def get_MACD(df, column='Adj Close'):
    """Return a new DataFrame with the MACD and related information (signal line and histogram)."""
    df['EMA-12'] = df[column].ewm(span=12, adjust=False).mean()
    df['EMA-26'] = df[column].ewm(span=26, adjust=False).mean()

    # MACD Indicator = 12-Period EMA − 26-Period EMA.
    df['MACD'] = df['EMA-12'] - df['EMA-26']

    # Signal line = 9-day EMA of the MACD line.
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Histogram = MACD - Indicator.
    df['Histogram'] = df['MACD'] - df['Signal']

    return df


# In[32]:


# this function add  column to data which are EMA-13	EMA-50	EMA-200
def get_remaining(df, column='Adj Close'):
    """Return a new DataFrame with the MACD and related information (signal line and histogram)."""
    df['EMA-13'] = df[column].ewm(span=13, adjust=False).mean()
    df['EMA-50'] = df[column].ewm(span=50, adjust=False).mean()
    df['EMA-200'] = df[column].ewm(span=200, adjust=False).mean()





    return df


# In[33]:


# this function add important column to data which are RSI	EMA-12	EMA-26	MACD	EMA-13	EMA-50	EMA-200	 ANS
def fun_create_dataset( df1 ):
    df1['RSI'] = computeRSI(df1['Adj Close'], 14)
    df1= get_MACD( df1 ).drop("Signal" , axis = 1).drop("Histogram" , axis = 1)
    df1= get_remaining( df1 )
    df1 = df1 [ df1 [ 'RSI' ]. isna() == False ]
    
    temp1 = df1["MACD"] < 0 
    temp1 = temp1.astype(int)
    temp1
    temp11 = df1["MACD"] >= 0 
    temp11 = temp11.astype(int)
    temp11
    
    temp2 =  df1["RSI"] > 60 
    temp2 = temp2.astype(int)
    temp2
    temp_21 =  df1["RSI"] > 60 
    temp_22 =  30 < df1["RSI"] 
    temp_2 = temp_21 & temp_22
    temp_2 = temp_2.astype(int)
    temp_2
    
    temp3 =  df1["EMA-200"] > df1["EMA-50"] 
    temp3 = temp3.astype(int)
    temp3
    temp_3 =  df1["EMA-200"] < df1["EMA-50"] 
    temp_3 = temp_3.astype(int)
    temp_3

    temp4 = df1["EMA-26"] > df1["EMA-13"] 
    temp4 = temp4.astype(int)
    temp4
    temp_4 = df1["EMA-26"] < df1["EMA-13"] 
    temp_4 = temp_4.astype(int)
    temp_4
    
    temp5 = temp1 + temp2 + temp3 + temp4
    temp5.unique()
    temp_5 = temp11 + temp_2 + temp_3 + temp_4
    temp_5.unique()
    
    
    temp6 = temp5>=3
    temp6 = temp6.astype(int)

    temp_5[temp_5<3]=0
    temp_5[temp_5>=3]=2

    temp_5.unique()

    ans = temp6 + temp_5

    df1["ANS"]=ans

    df1 = df1.loc["2000-01-01":].copy()
    df1
    return df1
    
    


# # function for feeding data and evaluate it

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier( random_state=1 , n_estimators = 100 , min_samples_split = 100 )

# this function use to feed data
def fun_feed( df1 ):
    x =df1[["Open" , "High" , "Low" , "Close" , "Adj Close" , "Volume"]]
    y = df1["ANS"]
    model.fit( x , y)
    
    


# In[35]:


from sklearn.metrics import precision_score

#this function use to test and evaluate
def fun_predict(df1):
    y_pred = model . predict( df1[["Open" , "High" , "Low" , "Close" , "Adj Close" , "Volume"]] )
    return precision_score(df1["ANS"] , y_pred, average='macro')
    
    
    
    


# converting dataset into dataframe

# In[36]:


df1 = fun_create_dataset( df1 )
df2 = fun_create_dataset( df2 )
df3 = fun_create_dataset( df3 )
df4 = fun_create_dataset( df4 )


# In[ ]:





# feeding train data to model : here model train on data of 3 different companies

# In[37]:


fun_feed( df1 )

fun_feed( df3 )

fun_feed( df4 )


# In[38]:





# testing and evaluating model on AAPLs stock : testing on AAPL's stock daat

# In[39]:


fun_predict( df2 )


# error is 0.24 which is less
