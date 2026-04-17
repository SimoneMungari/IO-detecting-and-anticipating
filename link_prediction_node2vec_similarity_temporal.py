import os.path

import torch
from torch_geometric.nn import Node2Vec
import sys
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pickle
import random
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    if seed is None:
        seed = 12121995
    print(f"[ Using Seed : {seed} ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    model,
    data_inter,
    data_sim,
    pos_edge_index,
    epochs=2000,
    lr=1e-2
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = data_inter.x
    edge_index_inter = data_inter.edge_index
    edge_index_sim = data_sim.edge_index

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        loss = link_prediction_loss(
            model,
            x,
            edge_index_inter,
            edge_index_sim,
            pos_edge_index
        )

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")


def reindex_graph(G_inter, G_sim):
    nodes = sorted(set(G_inter.nodes()) | set(G_sim.nodes()))
    node2idx = {n: i for i, n in enumerate(nodes)}

    def reindex(G):
        edges = []
        for u, v in G.edges():
            edges.append((node2idx[u], node2idx[v]))
        return edges

    return (
        nodes,
        node2idx,
        reindex(G_inter),
        reindex(G_sim),
    )

def predict(z, edge_index, sigmoid=False):
    # Dot-product decoder
    src, dst = edge_index
    if sigmoid:
        return (z[src] * z[dst]).sum(dim=1).sigmoid()
    else:
        return (z[src] * z[dst]).sum(dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = sys.argv[1]
similarity_network = sys.argv[2] if len(sys.argv) > 2 else "textSim"
threshold_to_cut = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

with open('data/processed/{}/datasets_full_temporal_{}.pkl'.format(dataset, threshold_to_cut), 'rb') as f:
    datasets = pickle.load(f)

print(datasets.keys())

if not os.path.exists("checkpoints/node2vec/{}_{}_{}_full_temporal".format(dataset, similarity_network, threshold_to_cut)):
    os.makedirs("checkpoints/node2vec/{}_{}_{}_full_temporal".format(dataset, similarity_network, threshold_to_cut))

print(similarity_network)

if dataset == "russia":
    embedding_dim = 32
    batch_size = 16
else:
    embedding_dim = 32
    batch_size = 256


G_inter = datasets['interaction_graph']
for node in G_inter.nodes():
    if not G_inter.has_edge(node, node):
        G_inter.add_edge(node, node)

G_sim = datasets[similarity_network]['train'].copy()

for node in G_sim.nodes():
    if not G_sim.has_edge(node, node):
        G_sim.add_edge(node, node)

for node in G_inter.nodes():
    if not G_sim.has_edge(node, node):
        G_sim.add_edge(node, node)

nodes, node2idx, inter_edges, sim_edges = reindex_graph(G_inter, G_sim)

test_ap = []
test_auc = []
seeds = [42, 45837, 92014, 18653, 60428]
for i in datasets['splits']:

    set_seed(seeds[i])

    best_model_path = "checkpoints/node2vec/{}_{}_{}_full_temporal/best_model_{}.pt".format(dataset, similarity_network, threshold_to_cut, i)

    pos_edge_index_train = sim_edges

    for node in G_sim.nodes():
        if (node2idx[node], node2idx[node]) not in pos_edge_index_train:
            pos_edge_index_train.append((node2idx[node], node2idx[node]))

    pos_edge_index_train = torch.tensor(pos_edge_index_train, dtype=torch.long).t().contiguous()

    pos_edge_index_val = []
    for edge in datasets['splits'][i]['val_positive_edges']:
        pos_edge_index_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    #exit(0)

    neg_edge_index_val = []
    for edge in datasets['splits'][i]['val_negative_edges']:
        neg_edge_index_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    pos_edge_index_val = torch.LongTensor(pos_edge_index_val).t().contiguous()
    neg_edge_index_val = torch.LongTensor(neg_edge_index_val).t().contiguous()

    pos_edge_index_test = []
    for edge in datasets['splits'][i]['test_positive_edges']:
        pos_edge_index_test.append((node2idx[edge[0]], node2idx[edge[1]]))

    neg_edge_index_test = []
    for edge in datasets['splits'][i]['test_negative_edges']:
        neg_edge_index_test.append((node2idx[edge[0]], node2idx[edge[1]]))

    pos_edge_index_test = torch.LongTensor(pos_edge_index_test).t().contiguous()
    neg_edge_index_test = torch.LongTensor(neg_edge_index_test).t().contiguous()

    # Model
    model = Node2Vec(
            pos_edge_index_train,
            embedding_dim=embedding_dim,
            walk_length=5,
            context_size=4,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(device)

    if dataset == "russia":
        num_workers = 1
    else:
        num_workers = 4
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)
    patience = 0
    max_patience = 30
    roc_auc_best = 0.0
    for epoch in range(5000):
        if patience > max_patience:
            break

        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 1 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                z = model()

                pos_logits = predict(z, pos_edge_index_val, sigmoid=True)
                neg_logits = predict(z, neg_edge_index_val, sigmoid=True)
                # print(pos_logits)

                pos_labels = torch.ones(pos_logits.size(0), device=z.device)
                neg_labels = torch.zeros(neg_logits.size(0), device=z.device)

                logits = torch.cat([pos_logits, neg_logits]).detach().cpu().numpy()
                labels = torch.cat([pos_labels, neg_labels]).detach().cpu().numpy()

                roc_auc = roc_auc_score(labels, logits)
                print(f"Node2Vec run {i}, epoch {epoch} Loss Train: {total_loss/len(loader):.4f}, VAL: AUC: {roc_auc:.4f}", flush=True)

                if roc_auc > roc_auc_best:
                    roc_auc_best = roc_auc
                    torch.save(model.state_dict(), best_model_path)
                    patience = 0
                else:
                    patience += 1

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        z = model()

        pos_logits = predict(z, pos_edge_index_test, sigmoid=True)
        neg_logits = predict(z, neg_edge_index_test, sigmoid=True)
        # print(pos_logits)

        pos_labels = torch.ones(pos_logits.size(0), device=z.device)
        neg_labels = torch.zeros(neg_logits.size(0), device=z.device)

        logits = torch.cat([pos_logits, neg_logits]).detach().cpu().numpy()
        labels = torch.cat([pos_labels, neg_labels]).detach().cpu().numpy()

        test_ap.append(average_precision_score(labels, logits))
        test_auc.append(roc_auc_score(labels, logits))

        print(f"Node2Vec Temporal run {i} AUC: {test_auc[-1]:.4f}, AP: {test_ap[-1]:.4f}", flush=True)

auc_mean = np.around(np.mean(test_auc), 3)
auc_std = np.around(np.std(test_auc), 3)

ap_mean = np.around(np.mean(test_ap), 3)
ap_std = np.around(np.std(test_ap), 3)

print(f"Node2Vec Temporal AUC: {auc_mean} +- {auc_std}, AP: {ap_mean} +- {ap_std}")

print("Finish", dataset)
