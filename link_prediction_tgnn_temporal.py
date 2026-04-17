from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.loader import LinkNeighborLoader, LinkLoader, TemporalDataLoader
from torch_geometric.nn import GCNConv, Node2Vec, SAGEConv, Linear, BatchNorm, TGNMemory, TransformerConv
import torch_geometric.transforms as T
from torch_geometric.data import Data, TemporalData
from torch_geometric.nn.models.tgn import LastAggregator, IdentityMessage, LastNeighborLoader, MeanAggregator
from torch_geometric.sampler import EdgeSamplerInput
from torch_geometric.utils import negative_sampling, add_remaining_self_loops
import sys
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pickle
import networkx as nx
import os
import random
import warnings
warnings.filterwarnings("ignore")

from utils import mergeNetworks

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


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels

        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

def link_prediction_loss(model, x, x_sim, edge_index_inter, edge_index_sim, pos_edge_index):
    z = model.encode(x, x_sim, edge_index_inter, edge_index_sim)

    # Positive edges
    pos_logits = model.compute_probability(z, edge_index_inter).reshape(-1)
    pos_labels = torch.ones(edge_index_inter.size(1), device=z.device)
    #print(pos_labels.shape, pos_logits.shape)

    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=edge_index_inter,
        num_nodes=z.size(0),
        num_neg_samples=edge_index_inter.size(1)
    )

    neg_logits = model.compute_probability(z, neg_edge_index).reshape(-1)
    neg_labels = torch.zeros(edge_index_inter.size(1), device=z.device)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])


    return F.binary_cross_entropy(logits, labels)

def train(
    memory,
    gnn,
    link_pred,
node2vec_int,
node2vec_sim,
    neighbor_loader,
    train_loader,
    train_data,
    pos_edge_index_val,
    neg_edge_index_val,
    assoc,
    optimizer,
    criterion,
    best_model_path_gnn,
    best_model_path_link_pred,
    best_model_path_memory,
    epochs=10000,
):
    patience = 0
    max_patience = 30
    val_roc_auc_best = 0

    for epoch in range(epochs):
        memory.train()
        gnn.train()
        link_pred.train()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, train_data.t[e_id],
                    train_data.msg[e_id])

            pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

            loss.backward()
            optimizer.step()

            memory.detach()
            total_loss = float(loss) * batch.num_events

        total_loss = total_loss / train_data.num_events

        if epoch % 10 == 0 and epoch > 0:

            with torch.no_grad():
                memory.eval()
                gnn.eval()
                link_pred.eval()

                torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

                aps, aucs = [], []

                z, last_update = memory(torch.arange(len(nodes)).to(device))

                z = gnn(z, last_update, train_data.edge_index, train_data.t,
                        train_data.msg)

                src, dst = pos_edge_index_val
                pos_out = link_pred(z[src], z[dst])
                src, dst = neg_edge_index_val
                neg_out = link_pred(z[src], z[dst])

                y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
                y_true = torch.cat(
                    [torch.ones(pos_out.size(0)),
                     torch.zeros(neg_out.size(0))], dim=0)

                aps.append(average_precision_score(y_true, y_pred))
                aucs.append(roc_auc_score(y_true, y_pred))

                aps_val = float(torch.tensor(aps).mean())
                val_roc_auc = float(torch.tensor(aucs).mean())

                if val_roc_auc > val_roc_auc_best:
                    patience = 0
                    val_roc_auc_best = val_roc_auc
                    torch.save(gnn.state_dict(), best_model_path_gnn)
                    torch.save(link_pred.state_dict(), best_model_path_link_pred)
                    torch.save(memory.state_dict(), best_model_path_memory)
                else:
                    patience += 1


                if patience >= max_patience:
                    print("Best", val_roc_auc_best)
                    return val_roc_auc_best

                print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | VAL: ROC-AUC {val_roc_auc:.4f}, AP {aps_val:.4f}", flush=True)

    print("Best", val_roc_auc_best)
    return val_roc_auc_best

def generate_node_features(G, nodes, latent_dim=32, feature_type="structural"):
    """
    Generate node features from NetworkX graph.
    """

    N = len(nodes)

    if feature_type == "positional_degree":
        def degree_to_one_hot(degree_dict, num_buckets, max_node_id):
            degrees_array = np.zeros((max_node_id, num_buckets), dtype=int)

            values = np.array(list(degree_dict.values()))
            percentiles = np.percentile(values, np.linspace(0, 100, num_buckets))

            for node_id, degree in degree_dict.items():
                bucket = np.searchsorted(percentiles, degree, side="right") - 1
                degrees_array[node2idx[node_id]][bucket] = 1

            return torch.FloatTensor(degrees_array)

        x = degree_to_one_hot(dict(nx.degree(G)), latent_dim, N)

    elif feature_type == "degree":
        deg = dict(G.degree())
        x = torch.tensor(
            [[deg.get(n, 0)] for n in nodes],
            dtype=torch.float
        )

    elif feature_type == "structural":
        deg = dict(G.degree())
        clustering = nx.clustering(G)

        x = torch.tensor(
            [
                [
                    deg.get(n, 0),
                    clustering.get(n, 0.0)
                ]
                for n in nodes
            ],
            dtype=torch.float
        )

    elif feature_type == "random":
        x = torch.randn(N, 32)

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return x


def nx_to_pyg_data(edges, num_nodes, x = None):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if x is None:
        x = torch.ones(num_nodes, 2)

    src, dst = edge_index

    return TemporalData(
        msg=torch.ones(len(edges)).reshape(-1, 1),
        t = torch.LongTensor(torch.arange(len(edges))),
        edge_index=edge_index,
        src = torch.LongTensor(src),
        dst = torch.LongTensor(dst)
    )


def align_graphs(G_inter, G_inter_train, G_inter_val, G_sim_train, G_sim_val):
    """
    Ensure same node ordering across graphs.
    """
    nodes = sorted(set(G_inter.nodes()) | set(G_inter_train.nodes()) | set(G_inter_val.nodes()) | set(G_sim_train.nodes()) | set(G_sim_val.nodes()))
    node2idx = {n: i for i, n in enumerate(nodes)}
    idx2node = {i: n for i, n in enumerate(nodes)}

    def reindex(G):
        edges = []
        for u, v in G.edges():
            edges.append((node2idx[u], node2idx[v]))
        return edges

    return (
        nodes,
        node2idx,
        idx2node,
        reindex(G_inter),
        reindex(G_inter_train),
        reindex(G_inter_val),
        reindex(G_sim_train),
        reindex(G_sim_val)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = sys.argv[1]
threshold_to_cut = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
hidden_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 32
out_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 8

latent_dim = 32

with open('data/processed/{}/datasets_full_temporal_{}.pkl'.format(dataset, threshold_to_cut), 'rb') as f:
    datasets = pickle.load(f)

if not os.path.exists("checkpoints/gnn/{}_full_temporal".format(dataset)):
    os.makedirs("checkpoints/gnn/{}_full_temporal".format(dataset))

G_inter = datasets['interaction_graph'].copy()
if dataset != "russia":
    G_inter_train = nx.from_edgelist(datasets['edges_train_gnn'][['userid', 'interaction_with_userid']].to_numpy())#datasets['interaction_graph_train'].copy()
else:
    G_inter_train = datasets['interaction_graph_train'].copy()

G_inter_val = nx.from_edgelist(datasets['edges_val_gnn'][['userid', 'interaction_with_userid']].to_numpy())
G_sim_train = datasets['final_graph']['train'].copy()
G_sim_val = datasets['final_graph']['val'].copy()

G_sim_train = G_sim_train.subgraph(G_inter.nodes())
G_sim_val = G_sim_val.subgraph(G_inter.nodes())


nodes, node2idx, idx2node, inter_edges, inter_edges_train, inter_edges_val, sim_edges_train, sim_edges_val = align_graphs(G_inter, G_inter_train, G_inter_val,
                                                                                      G_sim_train, G_sim_val)

if not os.path.exists("data/processed/{}/features_temporal/".format(dataset)):
    os.makedirs("data/processed/{}/features_temporal/".format(dataset))

feature_type = "positional_degree"

test_ap = []
test_auc = []
val_roc_auc_best_list = []
seeds = [42, 45837, 92014, 18653, 60428]

for i in datasets['splits']:

    set_seed(seeds[i])

    x_features_path = "data/processed/{}/features_temporal/G_inter_x_{}_{}_{}_temporal.pt".format(dataset, feature_type, i, latent_dim)
    x_features_path = "data/processed/{}/features_temporal/G_inter_x_full_{}_{}_{}_temporal.pt".format(dataset, feature_type, i, latent_dim)

    x_sim_train_features_path = "data/processed/{}/features_temporal/G_sim_train_x_{}_{}_{}_{}.pt".format(dataset, feature_type,
                                                                                           threshold_to_cut, i, latent_dim)
    x_sim_train_features_path = "data/processed/{}/features_temporal/G_sim_train_x_full_{}_{}_{}_{}.pt".format(dataset,
                                                                                                    feature_type,
                                                                                                    threshold_to_cut, i, latent_dim)

    x_sim_val_features_path = "data/processed/{}/features_temporal/G_sim_val_x_{}_{}_{}_{}.pt".format(dataset, feature_type,
                                                                                       threshold_to_cut, i, latent_dim)
    x_sim_val_features_path = "data/processed/{}/features_temporal/G_sim_val_x_full_{}_{}_{}_{}.pt".format(dataset, feature_type,
                                                                                                threshold_to_cut, i, latent_dim)

    x = generate_node_features(G_inter_train, nodes, feature_type=feature_type, latent_dim=latent_dim)
    torch.save(x, x_features_path)

    x_sim_train = generate_node_features(G_sim_train, nodes, feature_type=feature_type, latent_dim=latent_dim)
    torch.save(x_sim_train, x_sim_train_features_path)

    x_sim_val = generate_node_features(G_sim_val, nodes, feature_type=feature_type, latent_dim=latent_dim)
    torch.save(x_sim_val, x_sim_val_features_path)

    if not os.path.exists("checkpoints/gnn/{}_full_temporal/".format(dataset)):
        os.makedirs("checkpoints/gnn/{}_full_temporal/".format(dataset))

    if not os.path.exists("checkpoints/gnn/{}_temporal/".format(dataset)):
        os.makedirs("checkpoints/gnn/{}_temporal/".format(dataset))

    best_model_path_gnn = "checkpoints/gnn/{}_full_temporal/best_model_{}_{}_{}_{}_{}_{}_gnn.pt".format(dataset, i, feature_type, ablation, threshold_to_cut, hidden_dim, out_dim)
    best_model_path_link_pred = "checkpoints/gnn/{}_full_temporal/best_model_{}_{}_{}_{}_{}_{}_link_pred.pt".format(dataset, i,
                                                                                                        feature_type,
                                                                                                        ablation,
                                                                                                        threshold_to_cut,
                                                                                                        hidden_dim,
                                                                                                        out_dim)
    best_model_path_memory = "checkpoints/gnn/{}_full_temporal/best_model_{}_{}_{}_{}_{}_{}_memory.pt".format(dataset, i,
                                                                                                        feature_type,
                                                                                                        ablation,
                                                                                                        threshold_to_cut,
                                                                                                        hidden_dim,
                                                                                                        out_dim)

    transform = T.Compose([T.RemoveDuplicatedEdges()])

    data_inter_train = nx_to_pyg_data(inter_edges_train, len(nodes), x).to(device)
    data_inter_train = transform(data_inter_train)

    data_sim_train = nx_to_pyg_data(sim_edges_train, len(nodes), x_sim_train).to(device)
    data_sim_train = transform(data_sim_train)

    data_sim_val = nx_to_pyg_data(sim_edges_val, len(nodes), x_sim_val).to(device)
    data_sim_val = transform(data_sim_val)

    G_inter_val = nx.from_edgelist(list(G_inter_train.edges()) + list(G_inter_val.edges()))

    x_val = generate_node_features(G_inter_val, nodes, feature_type=feature_type, latent_dim=latent_dim)

    edges_val = []
    for edge in list(G_inter_val.edges()):
        edges_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    data_inter_val = nx_to_pyg_data(edges_val, len(nodes), x_val).to(device)
    data_inter_val = transform(data_inter_val)

    pos_edge_index_val = []
    for edge in datasets['splits'][i]['val_positive_edges']:
        pos_edge_index_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    pos_edge_index_val = torch.LongTensor(pos_edge_index_val).t().contiguous().to(device)

    neg_edge_index_val = []
    for edge in datasets['splits'][i]['val_negative_edges']:
        neg_edge_index_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    neg_edge_index_val = torch.LongTensor(neg_edge_index_val).t().contiguous().to(device)

    memory_dim = time_dim = embedding_dim = hidden_dim

    memory = TGNMemory(
        len(nodes),
        data_inter_train.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data_inter_train.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=out_dim,
        msg_dim=data_inter_train.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)


    link_pred = LinkPredictor(in_channels=out_dim).to(device)

    optimizer = torch.optim.AdamW(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    batch_size = 1024
    if dataset == "russia":
        batch_size = 200
    train_loader = TemporalDataLoader(
        data_inter_train,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
    )

    neighbor_loader = LastNeighborLoader(len(nodes), size=20, device=device)

    assoc = torch.empty(len(nodes), dtype=torch.long, device=device)

    val_roc_auc_best = train(memory, gnn, link_pred, None, None, neighbor_loader, train_loader, data_inter_train, pos_edge_index_val,
                             neg_edge_index_val, assoc, optimizer, criterion, best_model_path_gnn, best_model_path_link_pred, best_model_path_memory)
    val_roc_auc_best_list.append(val_roc_auc_best)

    gnn.load_state_dict(torch.load(best_model_path_gnn))
    link_pred.load_state_dict(torch.load(best_model_path_link_pred))
    memory.load_state_dict(torch.load(best_model_path_memory))
    # model.eval()

    with torch.no_grad():
        memory.eval()
        gnn.eval()
        link_pred.eval()

        pos_edge_index = []
        for edge in datasets['splits'][i]['test_positive_edges']:
            pos_edge_index.append((node2idx[edge[0]], node2idx[edge[1]]))
        # print(i, pos_edge_index)
        neg_edge_index = []
        for edge in datasets['splits'][i]['test_negative_edges']:
            neg_edge_index.append((node2idx[edge[0]], node2idx[edge[1]]))
        # print(i, neg_edge_index)

        pos_edge_index = torch.LongTensor(pos_edge_index).t().contiguous().to(device)
        neg_edge_index = torch.LongTensor(neg_edge_index).t().contiguous().to(device)

        # Update memory and neighbor loader with ground-truth state.
        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        memory.update_state(data_inter_val.src, data_inter_val.dst, data_inter_val.t, data_inter_val.msg)
        neighbor_loader.insert(data_inter_val.src, data_inter_val.dst)

        z, last_update = memory(torch.arange(len(nodes)).to(device))

        z = gnn(z, last_update, data_inter_val.edge_index, data_inter_val.t.to(device),
                data_inter_val.msg.to(device))

        src, dst = pos_edge_index
        pos_out = link_pred(z[src], z[dst])
        src, dst = neg_edge_index
        neg_out = link_pred(z[src], z[dst])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        print("Test ROC-AUC", roc_auc_score(y_true, y_pred), "AP", average_precision_score(y_true, y_pred), flush=True)
        test_ap.append(average_precision_score(y_true, y_pred))
        test_auc.append(roc_auc_score(y_true, y_pred))

        #break

print(val_roc_auc_best_list)
val_auc_mean = np.around(np.mean(val_roc_auc_best_list), 3)
val_auc_std = np.around(np.std(val_roc_auc_best_list), 3)

auc_mean = np.around(np.mean(test_auc), 3)
auc_std = np.around(np.std(test_auc), 3)

ap_mean = np.around(np.mean(test_ap), 3)
ap_std = np.around(np.std(test_ap), 3)

print(f"GNN VAL AUC: {val_auc_mean} +- {val_auc_std}")
print(f"GNN AUC: {auc_mean} +- {auc_std}, AP: {ap_mean} +- {ap_std}")

