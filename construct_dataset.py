import sys
import pickle
import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta
from utils import load_data

dataset = sys.argv[1]
if not os.path.exists('data/processed/{}'.format(dataset)):
    os.makedirs('data/processed/{}'.format(dataset))

mask_interactions = True if len(sys.argv) > 2 and sys.argv[2] == "1" else False

df = pd.read_pickle("data/{}/interactions_and_tweets_graph_full.pkl".format(dataset))
df_interactions = df[df['interaction_form'] != 'tweet']
df_interactions = df_interactions[df_interactions['interaction_type'] != "Unknown"].drop_duplicates(['userid', 'interaction_with_userid', 'tweetid'])

if mask_interactions:
    edges = df_interactions[['userid', 'interaction_with_userid']]
    node_labels = dict(df[['userid', 'group']].drop_duplicates().to_numpy())

    interaction_graph = nx.from_edgelist(edges.to_numpy())

    self_loops = list(nx.selfloop_edges(interaction_graph))
    interaction_graph.remove_edges_from(self_loops)

    isolated_nodes = list(nx.isolates(interaction_graph))
    interaction_graph.remove_nodes_from(isolated_nodes)

    datasets = {'coRetweet_graph': None,
                'coURL_graph': None,
                'textSim_graph': None,
                'fused_graph': None,
                'interaction_graph': interaction_graph, 'splits': {}}

    num_splits = 5
    seeds = [42, 45837, 92014, 18653, 60428]

    num_nodes = interaction_graph.number_of_nodes()
    num_edges = interaction_graph.number_of_edges()
    edges = np.array(list(interaction_graph.edges()))
    nodes = np.array(list(interaction_graph.nodes()))
    train_ratio = 0.6
    val_ratio = 0.2
    for i in range(num_splits):
        np.random.seed(seeds[i])

        indices = np.random.permutation(num_edges)

        n_train = int(train_ratio * num_edges)
        n_val = int(val_ratio * num_edges)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_mask = np.zeros(num_edges, dtype=bool)
        val_mask = np.zeros(num_edges, dtype=bool)
        test_mask = np.zeros(num_edges, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_positive_edges = edges[train_mask]
        train_negative_edges = np.concatenate((train_positive_edges[:, 0].reshape(-1, 1),
                                               nodes[np.random.choice(np.arange(num_nodes), size=len(train_positive_edges))].reshape(-1, 1)), axis=1)

        val_positive_edges = edges[val_mask]
        val_negative_edges = np.concatenate((val_positive_edges[:, 0].reshape(-1, 1),
                                               nodes[np.random.choice(np.arange(num_nodes),
                                                                      size=len(val_positive_edges))].reshape(-1, 1)),
                                              axis=1)

        test_positive_edges = edges[test_mask]
        test_negative_edges = np.concatenate((test_positive_edges[:, 0].reshape(-1, 1),
                                               nodes[np.random.choice(np.arange(num_nodes),
                                                                      size=len(test_positive_edges))].reshape(-1, 1)),
                                              axis=1)

        datasets['splits'][i] = {'train': train_mask, 'val': val_mask, 'test': test_mask,
                                 'train_positive_edges': train_positive_edges, 'train_negative_edges': train_negative_edges,
                                 'val_positive_edges': val_positive_edges, 'val_negative_edges': val_negative_edges,
                                 'test_positive_edges': test_positive_edges, 'test_negative_edges': test_negative_edges}


    path_output = 'data/processed/{}/datasets_full.pkl'.format(dataset)

    with open(path_output, 'wb') as handle:
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finish", dataset, "full", full, "mask interactions")
else:

    train_perc = 0.7
    val_perc = 0.15
    test_perc = 0.15

    df_interactions = df_interactions.sort_values('time')
    df_interactions = df_interactions.drop_duplicates(['userid', 'interaction_with_userid'], keep='last')

    timings = df_interactions['time'].sort_values()
    train_max_time = timings.iloc[:int(timings.shape[0]*train_perc)].max()
    val_max_time = timings.iloc[:int(timings.shape[0]*train_perc + timings.shape[0]*val_perc)].max()

    df_interactions_train = df_interactions[df_interactions['time'] < train_max_time]
    df_interactions_val = df_interactions[(df_interactions['time'] >= train_max_time) & (df_interactions['time'] < val_max_time)]
    df_interactions_test = df_interactions[df_interactions['time'] >= val_max_time]

    edges = df_interactions[['userid', 'interaction_with_userid']]

    edges_train = df_interactions_train[['userid', 'interaction_with_userid']]
    edges_train_gnn = df_interactions_train[['userid', 'interaction_with_userid', 'time']]
    print(edges_train.shape)
    edges_val = df_interactions_val[['userid', 'interaction_with_userid']]
    edges_val_gnn = df_interactions_val[['userid', 'interaction_with_userid', 'time']]
    print(edges_val.shape)
    edges_test = df_interactions_test[['userid', 'interaction_with_userid']]
    edges_test_gnn = df_interactions_test[['userid', 'interaction_with_userid', 'time']]
    print(edges_test.shape)

    node_labels = dict(df[['userid', 'group']].drop_duplicates().to_numpy())

    interaction_graph = nx.from_edgelist(edges.to_numpy())
    interaction_graph_train = nx.from_edgelist(edges_train.to_numpy())
    interaction_graph_val = nx.from_edgelist(edges_val.to_numpy())
    interaction_graph_test = nx.from_edgelist(edges_test.to_numpy())

    self_loops = list(nx.selfloop_edges(interaction_graph))
    interaction_graph.remove_edges_from(self_loops)
    isolated_nodes = list(nx.isolates(interaction_graph))
    interaction_graph.remove_nodes_from(isolated_nodes)

    self_loops = list(nx.selfloop_edges(interaction_graph_train))
    interaction_graph_train.remove_edges_from(self_loops)
    isolated_nodes = list(nx.isolates(interaction_graph_train))
    interaction_graph_train.remove_nodes_from(isolated_nodes)
    print(interaction_graph_train.number_of_edges())

    self_loops = list(nx.selfloop_edges(interaction_graph_val))
    interaction_graph_val.remove_edges_from(self_loops)
    isolated_nodes = list(nx.isolates(interaction_graph_val))
    interaction_graph_val.remove_nodes_from(isolated_nodes)
    print(interaction_graph_val.number_of_edges())

    self_loops = list(nx.selfloop_edges(interaction_graph_test))
    interaction_graph_test.remove_edges_from(self_loops)
    isolated_nodes = list(nx.isolates(interaction_graph_test))
    interaction_graph_test.remove_nodes_from(isolated_nodes)
    print(interaction_graph_test.number_of_edges())

    datasets = {'coRetweet_graph': None,
                'coURL_graph': None,
                'textSim_graph': None,
                'fused_graph': None,
                'train_max_time': train_max_time,
                'val_max_time': val_max_time,
                'interaction_graph': interaction_graph,
                'interaction_graph_train': interaction_graph_train,
                'edges_train_gnn': edges_train_gnn,
                'edges_val_gnn': edges_val_gnn,
                'edges_test_gnn': edges_test_gnn,
                'splits': {}}

    nodes = np.array(list(interaction_graph.nodes()))
    num_nodes = len(nodes)

    val_positive_edges = np.array(list(interaction_graph_val.edges()))
    test_positive_edges = np.array(list(interaction_graph_test.edges()))

    num_splits = 5
    seeds = [42, 45837, 92014, 18653, 60428]
    for i in range(num_splits):
        np.random.seed(seeds[i])

        val_negative_edges = np.concatenate((val_positive_edges[:, 0].reshape(-1, 1),
                                              nodes[np.random.choice(np.arange(num_nodes),
                                                                     size=len(val_positive_edges))].reshape(-1, 1)),
                                             axis=1)

        test_negative_edges = np.concatenate((test_positive_edges[:, 0].reshape(-1, 1),
                                              nodes[np.random.choice(np.arange(num_nodes),
                                                                     size=len(test_positive_edges))].reshape(-1, 1)),
                                             axis=1)
        datasets['splits'][i] = {
            'val_positive_edges': val_positive_edges, 'val_negative_edges': val_negative_edges,
            'test_positive_edges': test_positive_edges, 'test_negative_edges': test_negative_edges}

        print(len(val_positive_edges), len(val_negative_edges))


    path_output = 'data/processed/{}/datasets_full_temporal.pkl'.format(dataset)

    with open(path_output, 'wb') as handle:
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finish", dataset, "full", full)