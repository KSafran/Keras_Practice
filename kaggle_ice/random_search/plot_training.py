import pandas as pd
import seaborn as sns
import glob
import re
import matplotlib.pyplot as plt

all = pd.read_csv('data/exp_results.csv')

all['epoch'] = all.iloc[:,0]

g = sns.FacetGrid(all, col = 'exp_num', col_wrap = 5)
g = g.map(plt.plot, 'epoch', 'val_loss', color = 'blue')
g = g.map(plt.plot, 'epoch', 'loss', color = 'red')
g = g.set(ylim = (0, 1))
g.savefig('data/plot.png')


