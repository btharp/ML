#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# %%
df=pd.read_csv("nse-tata.csv")
df.head()
# %%
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

# %%
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

# %%
