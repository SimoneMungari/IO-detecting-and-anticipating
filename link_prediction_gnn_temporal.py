import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.loader import LinkNeighborLoader, LinkLoader
from torch_geometric.nn import GCNConv, Node2Vec, SAGEConv, Linear, BatchNorm
import torch_geometric.transforms as T
from torch_geometric.data import Data
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


class DualGraphLinkPredictor(nn.Module):
    def __init__(self, in_dim, in_dim_sim, hidden_dim, out_dim, edge_index_int, edge_index_sim, ablation="all", w=0.01):
        super().__init__()

        self.ablation = ablation
        self.w = w

        if ablation == "interactions":
            in_dim_sim = in_dim
        elif ablation == "similarities":
            in_dim = in_dim_sim

        if ablation == "all":
            self.gnn_interaction_1 = SAGEConv(in_dim*2, in_dim*2, root_weight=False)
        else:
            self.gnn_interaction_1 = SAGEConv(in_dim, in_dim * 2, root_weight=False)
        self.gnn_interaction_2 = SAGEConv(in_dim*2, in_dim*2, root_weight=False)
        self.gnn_interaction_3 = SAGEConv(in_dim * 2, in_dim * 2, root_weight=False)

        self.dropout_int = nn.Dropout(0.2)
        self.norm = LayerNorm(in_dim, elementwise_affine=True)
        self.norm_sim = LayerNorm(in_dim_sim, elementwise_affine=True)
        self.norm2 = LayerNorm(out_dim, elementwise_affine=True)

        # Similarity graph encoder
        if ablation == "all":
            self.gnn_similarity_1 = SAGEConv(in_dim_sim*2, in_dim_sim*2, root_weight=False)
        else:
            self.gnn_similarity_1 = SAGEConv(in_dim_sim, in_dim_sim * 2, root_weight=False)
        self.gnn_similarity_2 = SAGEConv(in_dim_sim*2, in_dim_sim*2, root_weight=False)
        self.gnn_similarity_3 = SAGEConv(in_dim_sim * 2, in_dim_sim * 2, root_weight=False)

        self.dropout_sim = nn.Dropout(0.2)

        self.activation_fn = torch.nn.ReLU()

        self.cross_attention_int = torch.nn.Linear(in_dim*2, hidden_dim)
        self.cross_attention_sim = torch.nn.Linear(in_dim_sim*2, hidden_dim)
        self.int_projector = torch.nn.Sequential(torch.nn.Linear(in_dim*2, hidden_dim),
                                                    torch.nn.ReLU())
        self.sim_projector = torch.nn.Sequential(torch.nn.Linear(in_dim_sim*2, hidden_dim),
                                                  torch.nn.ReLU())
        if ablation == "interactions":
            self.joint_projector = torch.nn.Sequential(
                torch.nn.Linear(in_dim * 2 + in_dim * 2, hidden_dim * 2),
                torch.nn.ReLU(),
                # nn.Linear(hidden_dim * 2, hidden_dim * 2),
                # nn.ReLU()
            )
        elif ablation == "similarities":
            self.joint_projector = torch.nn.Sequential(
                torch.nn.Linear(in_dim_sim * 2 + in_dim_sim * 2, hidden_dim * 2),
                torch.nn.ReLU(),
                # nn.Linear(hidden_dim * 2, hidden_dim * 2),
                # nn.ReLU()
            )
        else:
            self.joint_projector = torch.nn.Sequential(
                torch.nn.Linear(in_dim*2 + in_dim_sim * 2, hidden_dim * 2),
                torch.nn.ReLU(),
                # nn.Linear(hidden_dim * 2, hidden_dim * 2),
                # nn.ReLU()
            )

        self.final_layer = nn.Linear(in_dim_sim * 2, out_dim)

    def encode(self, x, x_sim, edge_index_inter, edge_index_sim):

        if self.ablation == "all":
            x = torch.concat([x, x_sim], dim=-1)
            x_sim = x

        if self.ablation in ["all", "interactions"]:
            h_i = self.gnn_interaction_1(x, edge_index_inter)
            h_i = self.activation_fn(h_i)
            h_i = self.dropout_int(h_i)
            h_i = self.gnn_interaction_2(h_i, edge_index_inter)
            h_i = self.activation_fn(h_i)
            h_i = self.dropout_int(h_i)
            h_i = self.gnn_interaction_3(h_i, edge_index_inter)

        if self.ablation in ["all", "similarities"]:
            h_s = self.gnn_similarity_1(x_sim, edge_index_sim)
            h_s = self.activation_fn(h_s)
            h_s = self.dropout_sim(h_s)
            h_s = self.gnn_similarity_2(h_s, edge_index_sim)
            h_s = self.activation_fn(h_s)
            h_s = self.dropout_sim(h_s)
            h_s = self.gnn_similarity_3(h_s, edge_index_sim)

        z_i = z_s = None
        if self.ablation == "all":

            z_i = self.int_projector(h_i)
            z_s = self.sim_projector(h_s)

            # Link prediction
            h = self.final_layer(h_i + h_s)

        elif self.ablation == "interactions":
            h = self.final_layer(h_i)
        elif self.ablation == "similarities":
            h = self.final_layer(h_s)

        return h, z_i, z_s

    def compute_probability(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1).sigmoid()


def contrastive_loss(z1, z2, temperature=0.2):
    """
    z1, z2: [N, D] embeddings for the same nodes from two views
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = z1 @ z2.T / temperature  # [N, N]
    labels = torch.arange(z1.size(0), device=z1.device)

    return F.cross_entropy(logits, labels)


def link_prediction_loss(model, x, x_sim, edge_index_inter, edge_index_sim, pos_edge_index):
    z, z_i, z_s = model.encode(x, x_sim, edge_index_inter, edge_index_sim)

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
    # print(neg_labels.shape, neg_logits.shape)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])

    # print("Train ROC-AUC", roc_auc_score(labels.detach().cpu(), logits.detach().cpu()),
    #       "AP", average_precision_score(labels.detach().cpu(), logits.detach().cpu()))
    if ablation == "all":
        cl_loss = contrastive_loss(z_i, z_s)

        loss = F.binary_cross_entropy(logits, labels) + model.w * cl_loss
    else:
        loss = F.binary_cross_entropy(logits, labels)

    return loss


def train(
    model,
    data_inter,
    data_sim,
    pos_edge_index,
    pos_edge_index_val,
    neg_edge_index_val,
    best_model_path,
    epochs=10000,
    lr=1e-3
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    patience = 0
    max_patience = 30
    val_roc_auc_best = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x = data_inter.x
        x_sim = data_sim.x
        edge_index_inter = data_inter.edge_index
        edge_index_sim = data_sim.edge_index


        loss = link_prediction_loss(
            model,
            x,
            x_sim,
            edge_index_inter,
            edge_index_sim,
            pos_edge_index
        )
        loss.backward()

        optimizer.step()

        if epoch % 10 == 0 and epoch > 0:
            model.eval()

            z, _, _ = model.encode(x, x_sim, edge_index_inter, edge_index_sim)

            # Positive edges
            pos_logits = model.compute_probability(z, pos_edge_index_val).reshape(-1)
            neg_logits = model.compute_probability(z, neg_edge_index_val).reshape(-1)
            # print(pos_logits)

            pos_labels = torch.ones(pos_logits.size(0), device=z.device)
            neg_labels = torch.zeros(neg_logits.size(0), device=z.device)

            logits = torch.cat([pos_logits, neg_logits]).detach().cpu().numpy()
            labels = torch.cat([pos_labels, neg_labels]).detach().cpu().numpy()

            val_roc_auc = roc_auc_score(labels, logits)

            if val_roc_auc > val_roc_auc_best:
                patience = 0
                val_roc_auc_best = val_roc_auc
                torch.save(model.state_dict(), best_model_path)
            else:
                patience += 1


            if patience >= max_patience:
                print("Best", val_roc_auc_best)
                return val_roc_auc_best

            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | VAL: ROC-AUC {val_roc_auc:.4f}, AP {average_precision_score(labels, logits):.4f}", flush=True)

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

    elif feature_type == "node2vec":
        node2vec = Node2Vec(
            G,
            embedding_dim=latent_dim,
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
            batch_size = 16
        else:
            num_workers = 4
            batch_size = 256
        loader = node2vec.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)
        for epoch in range(1000):
            node2vec.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
        x = node2vec()

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return x

def nx_to_pyg_data(edges, num_nodes, x = None):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if x is None:
        x = torch.ones(num_nodes, 2)

    return Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes
    )


def align_graphs(G_inter, G_inter_train, G_sim_train, G_sim_val):
    """
    Ensure same node ordering across graphs.
    """
    nodes = sorted(set(G_inter.nodes()) | set(G_inter_train.nodes()) | set(G_sim_train.nodes()) | set(G_sim_val.nodes()))
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
        reindex(G_sim_train),
        reindex(G_sim_val)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = sys.argv[1]
ablation = sys.argv[2] if len(sys.argv) > 2 else "all"
threshold_to_cut = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
hidden_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 32
out_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 8
w = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01

latent_dim = 32

with open('data/processed/{}/datasets_full_temporal_{}.pkl'.format(dataset, threshold_to_cut), 'rb') as f:
    datasets = pickle.load(f)

if not os.path.exists("checkpoints/gnn/{}_full_temporal".format(dataset)):
    os.makedirs("checkpoints/gnn/{}_full_temporal".format(dataset))

G_inter = datasets['interaction_graph'].copy()
G_inter_train = datasets['interaction_graph_train'].copy()
G_sim_train = datasets['final_graph']['train'].copy()
G_sim_val = datasets['final_graph']['val'].copy()

G_sim_train = G_sim_train.subgraph(G_inter.nodes())
G_sim_val = G_sim_val.subgraph(G_inter.nodes())

if ablation == "all":
    nodes, node2idx, idx2node, inter_edges, inter_edges_train, sim_edges_train, sim_edges_val = align_graphs(G_inter, G_inter_train, G_sim_train, G_sim_val)
elif ablation == "interactions":
    nodes, node2idx, idx2node, inter_edges, inter_edges_train, sim_edges_train, sim_edges_val = align_graphs(G_inter, G_inter_train,
                                                                                          G_sim_train, G_sim_val)
elif ablation == "similarities":
    nodes, node2idx, idx2node, inter_edges, inter_edges_train, sim_edges_train, sim_edges_val = align_graphs(G_inter, G_inter_train,
                                                                                          G_sim_train, G_sim_val)

if not os.path.exists("data/processed/{}/features_temporal/".format(dataset)):
    os.makedirs("data/processed/{}/features_temporal/".format(dataset))

feature_type = "node2vec"#"positional_degree"

test_ap = []
test_auc = []
val_roc_auc_best_list = []
seeds = [42, 45837, 92014, 18653, 60428]

for i in datasets['splits']:

    set_seed(seeds[i])

    x_features_path = "data/processed/{}/features_temporal/G_inter_x_{}_{}_{}_temporal.pt".format(dataset, feature_type, i, latent_dim)
    x_features_path = "data/processed/{}/features_temporal/G_inter_x_full_{}_{}_{}_temporal.pt".format(dataset, feature_type, i, latent_dim)

    x_val_features_path = "data/processed/{}/features_temporal/G_inter_val_x_{}_{}_{}_temporal.pt".format(dataset, feature_type,
                                                                                                  i, latent_dim)
    x_val_features_path = "data/processed/{}/features_temporal/G_inter_val_x_full_{}_{}_{}_temporal.pt".format(dataset,
                                                                                                           feature_type,
                                                                                                           i,
                                                                                                           latent_dim)

    x_sim_train_features_path = "data/processed/{}/features_temporal/G_sim_train_x_{}_{}_{}_{}.pt".format(dataset, feature_type,
                                                                                           threshold_to_cut, i, latent_dim)
    x_sim_train_features_path = "data/processed/{}/features_temporal/G_sim_train_x_full_{}_{}_{}_{}.pt".format(dataset,
                                                                                                    feature_type,
                                                                                                    threshold_to_cut, i, latent_dim)

    x_sim_val_features_path = "data/processed/{}/features_temporal/G_sim_val_x_{}_{}_{}_{}.pt".format(dataset, feature_type,
                                                                                       threshold_to_cut, i, latent_dim)
    x_sim_val_features_path = "data/processed/{}/features_temporal/G_sim_val_x_full_{}_{}_{}_{}.pt".format(dataset, feature_type,
                                                                                                threshold_to_cut, i, latent_dim)


    x = generate_node_features(G_inter_train, nodes, feature_type="random", latent_dim=latent_dim)
    torch.save(x, x_features_path)

    x_sim_train = generate_node_features(G_sim_train, nodes, feature_type="random", latent_dim=latent_dim)
    torch.save(x_sim_train, x_sim_train_features_path)

    x_sim_val = generate_node_features(G_sim_val, nodes, feature_type="random", latent_dim=latent_dim)
    torch.save(x_sim_val, x_sim_val_features_path)

    if not os.path.exists("checkpoints/gnn/{}_full_temporal/".format(dataset)):
        os.makedirs("checkpoints/gnn/{}_full_temporal/".format(dataset))

    if not os.path.exists("checkpoints/gnn/{}_temporal/".format(dataset)):
        os.makedirs("checkpoints/gnn/{}_temporal/".format(dataset))

    best_model_path = "checkpoints/gnn/{}_full_temporal/best_model_{}_{}_{}_{}_{}_{}_prova.pt".format(dataset, i, feature_type, ablation, threshold_to_cut, hidden_dim, out_dim)

    transform = T.Compose([T.ToUndirected(), T.AddRemainingSelfLoops(), T.RemoveDuplicatedEdges()])  # , T.AddRandomWalkPE(10)])

    data_inter_train = nx_to_pyg_data(inter_edges_train, len(nodes), x).to(device)
    data_inter_train = transform(data_inter_train)

    data_sim_train = nx_to_pyg_data(sim_edges_train, len(nodes), x_sim_train).to(device)
    data_sim_train = transform(data_sim_train)

    data_sim_val = nx_to_pyg_data(sim_edges_val, len(nodes), x_sim_val).to(device)
    data_sim_val = transform(data_sim_val)

    if not os.path.exists(x_features_path):
        x = generate_node_features(data_inter_train.edge_index, nodes, feature_type=feature_type, latent_dim=latent_dim)
        torch.save(x, x_features_path)
    else:
        x = torch.load(x_features_path)

    data_inter_train.x = x.to(device)

    if not os.path.exists(x_sim_train_features_path):
        x_sim_train = generate_node_features(data_sim_train.edge_index, nodes, feature_type=feature_type, latent_dim=latent_dim)
        torch.save(x_sim_train, x_sim_train_features_path)
    else:
        x_sim_train = torch.load(x_sim_train_features_path)

    data_sim_train.x = x.to(device)

    pos_edge_index_val = []
    for edge in datasets['splits'][i]['val_positive_edges']:
        pos_edge_index_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    pos_edge_index_val = torch.LongTensor(pos_edge_index_val).t().contiguous().to(device)

    neg_edge_index_val = []
    for edge in datasets['splits'][i]['val_negative_edges']:
        neg_edge_index_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    neg_edge_index_val = torch.LongTensor(neg_edge_index_val).t().contiguous().to(device)

    # Model
    model = DualGraphLinkPredictor(
        in_dim=data_inter_train.x.size(1),
        in_dim_sim=data_sim_train.x.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        edge_index_int=data_inter_train.edge_index,
        edge_index_sim=data_sim_train.edge_index,
        ablation=ablation,
        w = w
    ).to(device)

    val_roc_auc_best = train(model, data_inter_train, data_inter_train, data_inter_train.edge_index, pos_edge_index_val, neg_edge_index_val, best_model_path)
    val_roc_auc_best_list.append(val_roc_auc_best)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    G_inter_val = nx.from_edgelist(list(G_inter_train.edges()) + list(datasets['splits'][i]['val_positive_edges']))
    x_val = generate_node_features(G_inter_val, nodes, feature_type="random", latent_dim=latent_dim)

    edges_val = []
    for edge in list(G_inter_val.edges()):
        edges_val.append((node2idx[edge[0]], node2idx[edge[1]]))

    data_inter_val = nx_to_pyg_data(edges_val, len(nodes), x_val).to(device)
    data_inter_val = transform(data_inter_val)

    if not os.path.exists(x_val_features_path):
        x = generate_node_features(data_inter_val.edge_index, nodes, feature_type=feature_type,
                                   latent_dim=latent_dim)
        torch.save(x, x_val_features_path)
    else:
        x = torch.load(x_val_features_path)

    data_inter_val.x = x.to(device)

    with torch.no_grad():

        x = data_inter_val.x
        x_sim = data_sim_val.x
        edge_index_inter = data_inter_val.edge_index
        edge_index_sim = data_sim_val.edge_index

        pos_edge_index = []
        for edge in datasets['splits'][i]['test_positive_edges']:
            pos_edge_index.append((node2idx[edge[0]], node2idx[edge[1]]))

        neg_edge_index = []
        for edge in datasets['splits'][i]['test_negative_edges']:
            neg_edge_index.append((node2idx[edge[0]], node2idx[edge[1]]))

        pos_edge_index = torch.LongTensor(pos_edge_index).t().contiguous().to(device)
        neg_edge_index = torch.LongTensor(neg_edge_index).t().contiguous().to(device)

        z, _, _ = model.encode(x, x_sim, edge_index_inter, edge_index_sim)

        # Positive edges
        pos_logits = model.compute_probability(z, pos_edge_index).reshape(-1)
        neg_logits = model.compute_probability(z, neg_edge_index).reshape(-1)

        pos_labels = torch.ones(pos_logits.size(0), device=z.device)
        neg_labels = torch.zeros(neg_logits.size(0), device=z.device)

        logits = torch.cat([pos_logits, neg_logits]).detach().cpu().numpy()
        labels = torch.cat([pos_labels, neg_labels]).detach().cpu().numpy()

        print("Test ROC-AUC", roc_auc_score(labels, logits), "AP", average_precision_score(labels, logits), flush=True)
        test_ap.append(average_precision_score(labels, logits))
        test_auc.append(roc_auc_score(labels, logits))

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



