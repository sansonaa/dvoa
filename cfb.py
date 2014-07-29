###		CFB F/+ Analysis
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression as LR
from sklearn.neighbors import KNeighborsRegressor as KNReg
from sklearn.neighbors import RadiusNeighborsRegressor as RNReg
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_selection import f_regression

pd.set_option('display.max_columns', 120)


def evaluate_prediction(model,X,y):
	y_pred = model.predict(X)
	return "MAE: %.4f" % mae(y,y_pred), "MSE: %.4f" % mse(y,y_pred)

df = pd.read_csv('cfb_f_rankings.csv')

yrs = df.sort('Year').Year.unique()
tms = df.sort('Team').Team.unique()

# Break up Record into Wins, Losses, and WinPercentage
df['W'] = [int(r.split('-')[0]) for r in df.Record]
df['L'] = [int(r.split('-')[1]) for r in df.Record]
df['WinPerc'] = df.W/(df.L + df.W)

# Reformat f/+ values as percentages
effs = ['f', 'off_f', 'def_f', 'st_f']
for f in effs:
	df[f] = [v/100 for v in df[f]]

# Transform vertical (one row per school per year) to horizontal (one row per school, years across)
df_all = pd.DataFrame(data=tms, columns=['Team'])
keep_col = ['W', 'L', 'WinPerc', 'f', 'off_f', 'def_f', 'st_f', 's_p', 'fei']
for c in keep_col:
	rename = {}
	for y in yrs:
		rename[c] = "%d%s" % (y,c)
		df_all = pd.merge(df_all, df[df.Year == y][['Team', c]], how='left', on=['Team'], suffixes = ['', y])
		df_all.rename(columns=rename, inplace=True)

# Loop years, pulling n-year lags (to determine how consistent F/+ is season to season)
df_lag = pd.DataFrame(columns=['yr1_f','off_f','def_f','st_f','s_p','fei','yr2_f'])
for y in yrs:
	y2 = y+1
	if y2 not in yrs:
		break
	print y, "-", y2
	tms_include = np.intersect1d(df[df.Year == y].Team.values, df[df.Year == y2].Team.values)
	df_tmp = pd.merge(df[(df.Year == y) & (df.Team.isin(tms_include))][['Team','f','off_f','def_f','st_f','s_p','fei']],df[(df.Year == y2) & (df.Team.isin(tms_include))][['Team','f']],on=['Team'])
	df_tmp.rename(columns={'f_x':'yr1_f', 'f_y':'yr2_f'}, inplace=True)
	df_lag = df_lag.append(df_tmp[df_lag.columns])

# Calculate changes
# df_lag['change'] = df_lag.yr2 - df_lag.yr1
# df_lag['abs_change'] = abs(df_lag.yr2 - df_lag.yr1)
# for c in df_lag.columns:
	# df_lag[c] = df_lag[c].astype(float)

Xcol = ['yr1_f','off_f','def_f','st_f','s_p','fei']
ycol = ['yr2_f']
X_train, X_test, y_train, y_test = tts(df_lag[Xcol].values, df_lag[ycol].values)

linreg = LR()
linreg.fit(X_train, y_train)
linreg.score(X_train, y_train)



# Train on all existing seasons to project 2014
X,y = df_lag[Xcol].values, df_lag[ycol].values
linreg = LR()
linreg.fit(X, y)
linreg.score(X, y)

# build 3yr avgs
df_3avg = pd.DataFrame(columns=['avg_f']+['off_f','def_f','st_f','s_p','fei','yr4_f'])
for y in -np.sort(-yrs):
	
	if y2 not in yrs:
		break
	print y, "-", y2
	# tms_include = np.intersect1d(df[df.Year == y].Team.values, df[df.Year == y2].Team.values)
	# df_tmp = pd.merge(df[(df.Year == y) & (df.Team.isin(tms_include))][['Team','f']],df[(df.Year == y2) & (df.Team.isin(tms_include))][['Team','f']],on=['Team'])
	# df_tmp.rename(columns={'f_x':'yr1', 'f_y':'yr2'}, inplace=True)
	# df_lag = df_lag.append(df_tmp[df_lag.columns])

df2014 = df[df.Year == 2013][['Year', 'Team','f','off_f','def_f','st_f','s_p','fei']]
df2014['f2014'] = linreg.predict(df2014[['f','off_f','def_f','st_f','s_p','fei']].values)
df2014.sort('f2014', ascending=False,inplace=True)
df2014['rnk_2014'] = range(1,df2014.shape[0]+1)
#df2014.to_csv('f2014_project_oneyr.csv', index=False)


# build 3yr avgs
Xvar = ['f', 'off_f','def_f','st_f','s_p','fei']
##########
# previously ran and saved as 3yravg.csv
df_3avg = pd.read_csv('3yravg.csv', index_col='Team')
##########
# #df_3avg = pd.DataFrame(columns=Xvar)
# for y in -np.sort(-yrs):
	# y3 = [y-1,y-2,y-3]
	# if y3[2] not in yrs:
		# break
	# print y, ", ", y3
	# tms_include = np.intersect1d(df[df.Year == y3[0]].Team.values, df[df.Year == y3[2]].Team.values)
	# df_tmp = pd.merge(df[(df.Year.isin(y3)) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), df[(df.Year == y3[0]) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), how='left',left_index=True, right_index=True, suffixes=['_3yr_avg','_yr3'])
	# # df_tmp = pd.merge(df[(df.Year.isin(y3)) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), df[(df.Year == y) & (df.Team.isin(tms_include))][['Team','f']], how='left',left_index=True, right_on='Team')

	# df_tmp2 = df[(df.Year == y) & (df.Team.isin(tms_include))].groupby('Team')['f'].mean()
	# df_tmp2 = pd.DataFrame(df_tmp2)
	# df_tmp = pd.merge(df_tmp, df_tmp2, how='left',left_index=True, right_index=True)
	# df_tmp.rename(columns={'f':'yr4_f'}, inplace=True)
	# if df_3avg is None:
		# df_3avg = df_tmp
	# else:
		# df_3avg = df_3avg.append(df_tmp)

#Regress 3 yr avgs and project 2014 based on 2011,12, and 13
threeYrXcol = df_3avg.columns[:-1]
threeYrycol = df_3avg.columns[-1:]

#Feature Selection for 3 year avgs/prev yr
F,p = f_regression(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)
for k, v in enumerate(p):
	print threeYrXcol[k], ": ", v

# Break up into TTS slices
X_train, X_test, y_train, y_test = tts(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)

# Linear Regression
lin3 = LR()
#lin3.fit(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)
lin3.fit(X_train, y_train)
print "Train: ", lin3.score(X_train, y_train)
print "Test: ", lin3.score(X_test, y_test)
print "Intercept: ", lin3.intercept_
for k, v in enumerate(lin3.coef_[0]):
	print threeYrXcol[k], ": ", v

# KNeighborsRegressor
kn3 = KNReg(weights='uniform')
#kn3.fit(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)
kn3.fit(X_train, y_train)
print "Train: ", kn3.score(X_train, y_train)
print "Test: ", kn3.score(X_test, y_test)
# print kn3.score(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)

# RadiusNeighborsRegressor
rn3 = RNReg(radius=7.0)
#rn3.fit(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)
rn3.fit(X_train, y_train)
print "Train: ", rn3.score(X_train, y_train)
print "Test: ", rn3.score(X_test, y_test)
print rn3.score(df_3avg[threeYrXcol].values, df_3avg[threeYrycol].values)

# Test 2010/11/12 stats and 2013 projections against 2013 actuals
y=2013
y3 = [y-1,y-2,y-3]
tms_include = np.intersect1d(df[df.Year == y3[0]].Team.values, df[df.Year == y3[2]].Team.values)
df2012 = pd.merge(df[(df.Year.isin(y3)) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), df[(df.Year == y3[0]) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), how='left',left_index=True, right_index=True, suffixes=['_3yr_avg','_yr3'])
df2012['f2013'] = lin3.predict(df2012.values)
df2012.sort('f_yr3', ascending=False, inplace=True)
df2012['rnk_2012'] = range(1,df2012.shape[0]+1)
df2012.sort('f2013', ascending=False, inplace=True)
df2012['rnk_2013'] = range(1,df2012.shape[0]+1)
#df2012.to_csv('f2013_projection_3yrs.csv', headers=True,index=True)

##########
### PROJECTIONS - 2014
##########
# Get 2011/12/13 stats for 2014 projection
y=2014
y3 = [y-1,y-2,y-3]
tms_include = np.intersect1d(df[df.Year == y3[0]].Team.values, df[df.Year == y3[2]].Team.values)
df2013 = pd.merge(df[(df.Year.isin(y3)) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), df[(df.Year == y3[0]) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), how='left',left_index=True, right_index=True, suffixes=['_3yr_avg','_yr3'])
df2013['f2014'] = lin3.predict(df2013.values)
df2013.sort('f_yr3', ascending=False, inplace=True)
df2013['rnk_2013'] = range(1,df2013.shape[0]+1)
df2013.sort('f2014', ascending=False, inplace=True)
df2013['rnk_2014'] = range(1,df2013.shape[0]+1)
#df2013.to_csv('f2014_projection_3yrs.csv', headers=True,index=True)

# Get 2011/12/13 stats for 2014 projection
y=2014
y3 = [y-1,y-2,y-3]
tms_include = np.intersect1d(df[df.Year == y3[0]].Team.values, df[df.Year == y3[2]].Team.values)
df2013 = pd.merge(df[(df.Year.isin(y3)) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), df[(df.Year == y3[0]) & (df.Team.isin(tms_include))].groupby('Team')[Xvar].mean(), how='left',left_index=True, right_index=True, suffixes=['_3yr_avg','_yr3'])
df2013['f2014'] = rn3.predict(df2013.values)
df2013.sort('f_yr3', ascending=False, inplace=True)
df2013['rnk_2013'] = range(1,df2013.shape[0]+1)
df2013.sort('f2014', ascending=False, inplace=True)
df2013['rnk_2014'] = range(1,df2013.shape[0]+1)
df2013.to_csv('f2014_projection_3yrs_rn.csv', headers=True,index=True)