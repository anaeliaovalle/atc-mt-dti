import pandas as pd
import numpy as np
import networkx as nx

adj = pd.read_csv("/Users/yuyan/Desktop/Papers/Course/advanced data mining/project/atc-mt-dti/data/AdjacencyMatrix_AllATC.csv")

adj = adj.iloc[:,1:len(adj.columns)]
adj = adj.to_numpy()

D = nx.DiGraph(adj)
D = D.to_undirected()
nx.write_weighted_edgelist(D, '/Users/yuyan/Desktop/Papers/Course/advanced data mining/project/atc-mt-dti/data/ATC_edgelist.txt')
