import torch 
import numpy as np
import pickle

model = torch.load("model.pt")
emb = model.nodes_embeddings.weight.detach().numpy()

with open("/Users/yuyan/Desktop/Papers/Course/advanced data mining/project/atc-mt-dti/data/ACT_embedding.pdl",'wb') as f:
    pickle.dump(emb,f)
