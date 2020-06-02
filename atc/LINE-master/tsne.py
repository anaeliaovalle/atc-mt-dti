import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

dat = pickle.load(open('../data/ATC_embedding.pkl','rb'))
with open("../data/drug_name.txt") as f:
    drug_name = f.readline().strip().split(',')


tsne = manifold.TSNE()
y = tsne.fit_transform(dat)

fig, ax = plt.subplots()
ax.scatter(y[:,0],y[:,1])

#for i, txt in enumerate(drug_name):
#    ax.annotate(txt, (y[i,0],y[i,1]))

plt.savefig('../data/ATCtsne.pdf')
