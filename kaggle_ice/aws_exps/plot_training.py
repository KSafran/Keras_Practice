import pandas as pd
import seaborn as sns
import glob
import re
import matplotlib.pyplot as plt

files = glob.glob('data/*.csv')

dfs = []
for a in files:
	df = pd.read_csv(a)
	df['optimizer'] = re.search('rms|adam', a).group()
	df['learn_rate'] = re.search('[0-2]', a).group()
	dfs.append(df)

all = pd.concat(dfs)
all.columns.values[0] = 'epoch'

g = sns.FacetGrid(all, col = 'learn_rate', row = 'optimizer')
g = g.map(plt.plot, 'epoch', 'val_loss', color = 'red')
g = g.map(plt.plot, 'epoch', 'loss', color = 'blue')
g = g.set(ylim = (0, 1))
g.savefig('data/plot.png')


