import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from matplotlib.ticker import NullFormatter
from matplotlib import colors as mcolors

from sklearn import manifold, datasets
from collections import defaultdict, Counter

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]
sorted_names = random.sample(sorted_names, len(sorted_names))


dat = pickle.load(open('../data/ATC_embedding.pkl','rb'))
with open("../data/drug_name.txt") as f:
    drug_name = f.readline().strip().split(',')

ATC_class = []
clss = Counter()
with open('../data/ATC_class.txt') as f:
    for line in f:
        ATC_class.append(line.strip())
        clss[line.strip()] +=1

cls2dict = {}
for i,v in zip(sorted_names[0:len(clss)],list(clss.keys())):
    cls2dict[v] = i

ATC_col = [cls2dict[c] for c in ATC_class]



tsne = manifold.TSNE()
y = tsne.fit_transform(dat)

fig, ax = plt.subplots()
ax.scatter(y[:,0], y[:,1], c = ATC_col)
#ax.legend()



#for i, txt in enumerate(drug_name):
#    ax.annotate(txt, (y[i,0],y[i,1]))

plt.savefig('../data/ATCtsne.pdf')


# add label
fig, ax = plt.subplots()
for l, c in zip(list(cls2dict.keys()), [cls2dict[cs] for cs in list(cls2dict.keys())]):
    ax.scatter(1,1,c = c, label = l)

ax.legend()
plt.show()

