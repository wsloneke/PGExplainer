import sys
import os
from scipy.sparse import load_npz
sys.path.append('./codes/forgraph/')
from config import args
from sklearn.metrics import roc_auc_score
from models import GCN
from metrics import *
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix,csr_matrix
import pickle as pkl
import csv
import torch
from Explainer import Explainer


def acc(adj,insert):
    mask = explainer.masked_adj.numpy()
    adj = coo_matrix(adj)
    for r,c in list(zip(adj.row,adj.col)):
        if r>=insert and r<insert+skip and c>=insert and c<insert+skip:
            reals.append(1)
        else:
            reals.append(0)
        preds.append(mask[r][c])
        
def explain_graph(gid):
    fea,emb,adj,label,graphid = features[gid], embs[gid], adjs[gid], labels[gid], gid
    explainer((fea,emb,adj,1.0,label))
    insert = 20
    acc(adj,insert)
    
def test():
    global preds
    global reals
    preds = []
    reals = []
    for gid in allnodes:
        explain_graph(gid)
    auc = roc_auc_score(reals,preds)
    subset_acc = auc
    all_node_acc = roc_auc_score(reals, allnodes)
    fidelity = all_node_acc - subset_acc
    subset_nodes = len(preds)
    all_nodes = len(allnodes)
    sparsity = 1 - subset_nodes / all_nodes
    return auc, fidelity, sparsity

def train(epochs):
    t0 = args.coff_t0
    t1 = args.coff_te

    for epoch in range(epochs): 
        loss = 0
        tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
        train_instances = [ins for ins in range(len(model.values()))]
        np.random.shuffle(train_instances)
        for gid in train_instances:
            with tf.GradientTape() as tape:
                pred = explainer((features[gid],embs[gid],adjs[gid],tmp, labels[gid]),training=True)
                loss += explainer.loss(pred, labels[gid])
        train_variables = [para for para in explainer.trainable_variables]
        grads = tape.gradient(loss, train_variables)
        optimizer.apply_gradients(zip(grads, train_variables))
        
cell_line = 'E116'
base_path = 'actual data/src/data'
hic_sparse_mat_file = os.path.join(base_path, cell_line, 'hic_sparse.npz')
np_nodes_lab_genes_file = os.path.join(base_path, cell_line, 'np_nodes_lab_genes.npy')
np_hmods_norm_all_file = os.path.join(base_path, cell_line, 'np_hmods_norm.npy') 
df_genes_file = os.path.join(base_path, cell_line, 'df_genes.pkl')
df_genes = pd.read_pickle(df_genes_file)

mat = load_npz(hic_sparse_mat_file)
allNodes_hms = np.load(np_hmods_norm_all_file)
hms = allNodes_hms[:, 1:] #only includes features, not node ids
allNodes = allNodes_hms[:, 0].astype(int)
geneNodes_labs = np.load(np_nodes_lab_genes_file)
geneNodes = geneNodes_labs[:, -2].astype(int)
geneLabs = geneNodes_labs[:, -1].astype(int)

allLabs = 2*np.ones(np.shape(allNodes))
allLabs[geneNodes] = geneLabs

features = hms
adjs = mat
labels = allLabs

all_nodes = len(labels)

embs = scipy.sparse.hstack([features, adjs])

file = 'actual data/src/data/E116/saved_runs/model_2020-12-17-at-19-24-36.pt'
model = torch.load(file, map_location=torch.device('cpu'))

top_k = [i for i in range(0, 26)]
all_elr = [i for i in range(0, 26)]
coff_t0 = [i for i in range(1, 26)]
coff_size = [i for i in range(0, 26)]
coff_ent = [i for i in range(0, 26)]
skp = [i for i in range(0, 26)]
epochs = [i for i in range(250, 650, 50)]

rows = []

explainer = Explainer(model=model,nodesize=all_nodes)

values = embs.data
indices = np.vstack((embs.row, embs.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = embs.shape

embs = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

for k in top_k:
    topk = k
    for elr in all_elr:
        args.elr = elr
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.elr)
        for c in coff_t0:
            args.coff_t0 = c
            for s in coff_size:
                args.coff_size = s
                for en in coff_ent:
                    args.coff_ent = en
                    for sk in skp:
                        skip = sk
                        for ep in epochs:
                            train(ep)
                            auc, fidelity, sparsity = test()
                            row = [k, elr, c, en, sk, ep, auc, fidelity, sparsity]
                            rows.append(row)
                            
col_names = ['K', 'elr', 'coff_t0', 'coff_size', 'coff_ent', 'Skip', 'AUC', 'Fidelity', 'Sparsity']
filename = 'results.csv'
with open('results.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(col_names) 
    write.writerows(rows) 