###		NFL Down/Distance by Year (1989-2014)

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot as plt

from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression as LR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.neighbors import RadiusNeighborsRegressor as RNR
from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import auc

pd.set_option('display.max_columns', 120)

def evaluate_prediction(model,X,y):
	y_pred = model.predict(X)
	return "MAE: %.4f" % mae(y,y_pred), "MSE: %.4f" % mse(y,y_pred)

	
df = pd.read_csv('down_dist.csv', index_col='Tm')

yrs = df.Yr.unique()
Xcol = [c for c in df.columns if c != "Yr"]
Xcol = ['1D', '2D', '3D', 'all']

df_avg = None
rolling_yrs = 3
for y in yrs:
	prevYr = y-1
	yr1 = y - rolling_yrs
	if yr1 not in yrs:
		break
	tms_include = np.intersect1d(df[df.Yr == y].index.values, df[df.Yr == yr1].index.values)
	df_tmp = df[(df.index.isin(tms_include)) & (df.Yr < y) & (df.Yr >= yr1)].groupby(df[(df.index.isin(tms_include)) & (df.Yr < y) & (df.Yr >= yr1)].index)[Xcol].mean()
	df_tmp = pd.merge(df[(df.index.isin(tms_include)) & (df.Yr == y)][Xcol], df_tmp, how='left', left_index=True, right_index=True, suffixes=('_yr', '_avg'))
	df_tmp = pd.merge(df_tmp, df[(df.index.isin(tms_include)) & (df.Yr == prevYr)][Xcol], how='left', left_index=True, right_index=True, suffixes=('', '_prev'))
	if df_avg is None:
		df_avg = df_tmp
	else:
		df_avg = df_avg.append(df_tmp)

#df_avg.drop('Yr_prev', 1, inplace=True)
for x in Xcol:
	x_new = "%s_prev" % x
	df_avg.rename(columns={x:x_new}, inplace=True)

X,y = df_avg[['3D_avg', 'all_avg','3D_prev', 'all_prev']].values, df_avg['all_yr'].values

#f_regression
F_avg, p_avg = f_regression(df_avg[['all_avg']].values, df_avg['all_yr'].values)
F_prev, p_prev = f_regression(df_avg[['all_prev']].values, df_avg['all_yr'].values)	
F_comb, p_comb = f_regression(df_avg[['all_prev', 'all_avg']].values, df_avg['all_yr'].values)	

print "Avg: ", p_avg
print "Prev: ", p_prev
print "Comb: "
print " -- prev", p_comb[0]
print " -- avg", p_comb[1]

linAvg = LR()
linAvg.fit(df_avg[['all_avg']].values, df_avg['all_yr'].values)
print linAvg.score(df_avg[['all_avg']].values, df_avg['all_yr'].values)

linPrev = LR()
linPrev.fit(df_avg[['all_prev']].values, df_avg['all_yr'].values)
print linPrev.score(df_avg[['all_prev']].values, df_avg['all_yr'].values)
	
linComb = LR()
linComb.fit(df_avg[['all_prev', 'all_avg']].values, df_avg['all_yr'].values)
print linComb.score(df_avg[['all_prev', 'all_avg']].values, df_avg['all_yr'].values)

linDet = LR()
linDet.fit(X,y)
print linDet.score(X,y)

# df_avg == 3 year rolling average + yr4 stats
X,y = df_avg[['all_avg']].values, df_avg['all_yr'].values
X,y = df_avg[['all_prev']].values, df_avg['all_yr'].values
X,y = df_avg[['all_avg', 'all_prev']].values, df_avg['all_yr'].values

X,y = df_avg[['1D_avg', '2D_avg', '3D_avg', 'all_avg','1D_prev', '2D_prev', '3D_prev', 'all_prev']].values, df_avg['all_yr'].values
X_train, X_test, y_train, y_test = tts(X, y)

lin = LR(fit_intercept=False)
lin.fit(X,y)
lin.score(X,y)

knn = KNR(n_neighbors=5)
knn.fit(X_train,y_train)
print knn.score(X_train,y_train)
print knn.score(X_test,y_test)


ns = range(1,30,2)
scores = []
for n in ns:
	knn = KNR(n_neighbors=n)
	knn.fit(X_train,y_train)
	scores.append(knn.score(X_train,y_train))


rf = RFR(n_estimators = 50)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)


# Loop and Test
scores = []
leaves = range(1,15,2)
depth = range(1,30,2)
for d in depth:
	rf = RFR(n_estimators = 100, oob_score=True, max_depth=d, max_features ='sqrt', n_jobs=-1)
	rf.fit(X_train, y_train)
	scores.append(rf.score(X_train, y_train))
plt.plot(depth, scores)
plt.show()


rf = RFR(n_estimators = 50, oob_score=True, max_depth=15, max_features ='sqrt', n_jobs=-1)
rf.fit(X_train, y_train)
print rf.score(X_train, y_train)
print rf.score(X_test, y_test)


# Loop and Test n Trees
trees = range(100,5101,500)
scores = []
for t in trees:
	rf = RFR(n_estimators = t, oob_score=True, max_depth=15, max_features ='sqrt', n_jobs=-1)
	rf.fit(X_train, y_train)
	#scores.append(rf.score(X_train, y_train))
	scores.append(rf.score(X_test, y_test))
	
plt.plot(trees, scores)
plt.show()










scatter_col = ['1D_avg', '2D_avg', '3D_avg', 'all_avg', '1D_yr4', '2D_yr4', '3D_yr4', 'all_yr4','1D_yr3', '2D_yr3', '3D_yr3', 'all_yr3']
scatter_matrix(df_avg[scatter_col], alpha=0.2, figsize=(6, 6), diagonal='kde')