import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import snscrape.modules.twitter as sntwitter
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

old_df = pd.read_csv("Twitter_stock_final_dataset.csv")
old_df["Date"] = pd.to_datetime(old_df[['Day','Month','Year']])
old_df.index=old_df.Date

le=LabelEncoder()
le1=LabelEncoder()

old_df.StockName = le.fit_transform(old_df.StockName)
old_df.Year = le1.fit_transform(old_df.Year)

sc1 = StandardScaler()
sc2 = StandardScaler()
old_df.iloc[:,9] = sc1.fit_transform(np.array(old_df.iloc[:,9]).reshape(-1,1))
old_df.iloc[:,8] = sc2.fit_transform(np.array(old_df.iloc[:,8]).reshape(-1,1))

old_df = old_df[["Year","StockName","Positive","Negative","Neutral","Close","Volume"]]
d = pd.DataFrame()
d = pd.get_dummies(old_df.StockName, prefix=None, prefix_sep='_', dummy_na=False)
old_df1 = pd.concat([old_df,d], axis=1)
old_df1.drop(['StockName'], axis=1, inplace=True)

d = pd.DataFrame()
d = pd.get_dummies(old_df1.Year, prefix=None, prefix_sep='_', dummy_na=False)
old_df1 = pd.concat([old_df1,d], axis=1)
old_df1.drop(['Year'], axis=1, inplace=True)

X = np.array(old_df1.drop(["Close"],1))
y = np.array(old_df1.Close)

from sklearn.ensemble import RandomForestRegressor
rf_2 = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=3, min_samples_split=8, n_estimators=1000, random_state=1)
rf_2.fit(X,y)

pickle.dump(rf_2, open('first.pkl', 'wb'))



