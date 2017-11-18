import pandas as pd
import seaborn as sns
import glob
import re
import matplotlib.pyplot as plt

files = glob.glob('data/*.csv')

dfs = []
for a in files:
	df = pd.read_csv(a)
	df['channel'] = re.search('avg|diff|none', a).group()
	
	if re.search('norm', a) and re.search('angle', a):
		df['options'] = 'both'
	elif re.search('norm', a):
		df['options'] = 'norm'
	elif re.search('angle', a):
		df['options'] = 'angle'
	else:
		df['options'] = 'none'
	dfs.append(df)

all = pd.concat(dfs)
all.columns.values[0] = 'epoch'

g = sns.FacetGrid(all, col = 'channel', row = 'options')
g = g.map(plt.plot, 'epoch', 'val_loss', color = 'blue')
g = g.map(plt.plot, 'epoch', 'loss', color = 'red')
g = g.set(ylim = (0, 1))
g.savefig('data/plot.png')


