###		CFB F/+ Analysis
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 120)

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

for y in yrs:
	y2 = y+1
	if y2 not in yrs:
		break
	print y, "-", y2
	tms_include = np.intersect1d(df[df.Year == y].Team.values, df[df.Year == y2].Team.values)
	oneYr = df[df.Year == y].sort('Team')['f'].values,df[df.Year == y2].sort('Team')['f'].values
	fld1, fld2 = "%df" % y, "%df" % y2
	oneYr = df_all[fld1]

