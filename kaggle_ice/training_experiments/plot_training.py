import pandas as pd
import seaborn as sns
import glob
import re
import matplotlib.pyplot as plt

files = glob.glob('data/*.csv')

dfs = []
for a in files:
	dfs.append(pd.read_csv(a))

all = pd.concat(dfs)
all['epoch'] = all.index

g = sns.FacetGrid(all, col = 'learn', row = 'batch_size')
g = g.map(plt.plot, 'epoch', 'val_loss', color = 'blue')
g = g.map(plt.plot, 'epoch', 'loss', color = 'red')
g = g.set(ylim = (0, 1))
g.savefig('data/plot.png')


