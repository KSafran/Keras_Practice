import pandas as pd
import seaborn as sns
import glob
import re
import matplotlib.pyplot as plt

dfs = []
for a in range(11, 19,1):
	dfs.append(pd.read_csv('data/hist_' + str(a) + '.csv'))

all = pd.concat(dfs)
all['epoch'] = all.index

g = sns.FacetGrid(all, col = 'learn', row = 'optim')
g = g.map(plt.plot, 'epoch', 'val_loss', color = 'blue')
g = g.map(plt.plot, 'epoch', 'loss', color = 'red')
g = g.set(ylim = (0, 1))
g.savefig('data/plot_2.png')


