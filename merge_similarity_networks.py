import os.path
import sys
import networkx as nx
from utils import mergeNetworks
import pickle
import numpy as np

dataset = sys.argv[1]
threshold_to_cut = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
temporal = True if len(sys.argv) > 3 and sys.argv[3] == "1" else False


if not temporal:
    with open('data/processed/{}/datasets_full.pkl'.format(dataset), 'rb') as f:
        datasets = pickle.load(f)

else:
    with open('data/processed/{}/datasets_full_temporal.pkl'.format(dataset), 'rb') as f:
        datasets = pickle.load(f)

similarity_networks = ["coRetweet_graph", "coURL_graph", "textSim_graph", "final_graph"]

if not temporal:
    for split in datasets['splits']:
        for similarity_network in similarity_networks:
            print(similarity_network, split)
            datasets['splits'][split][similarity_network] = {'train': None, 'val': None}

            if similarity_network == "final_graph":
                datasets['splits'][split][similarity_network]['train'] = mergeNetworks(
                    [datasets['splits'][split]["coRetweet_graph"]['train'],
                     datasets['splits'][split]["coURL_graph"]['train'],
                     datasets['splits'][split]["textSim_graph"]['train']])
                nx.write_gml(datasets['splits'][split]['final_graph']['train'],
                             "data/{}/similarity_networks/final_graph_full_mask_interactions_train_{}.gml".format(dataset,
                                                                                                                split))

                datasets['splits'][split][similarity_network]['val'] = mergeNetworks(
                    [datasets['splits'][split]["coRetweet_graph"]['val'],
                     datasets['splits'][split]["coURL_graph"]['val'],
                     datasets['splits'][split]["textSim_graph"]['val']])
                nx.write_gml(datasets['splits'][split]['final_graph']['val'],
                             "data/{}/similarity_networks/final_graph_full_mask_interactions_val_{}.gml".format(dataset, split))

            datasets['splits'][split][similarity_network]['train'] = nx.read_gml(
                "data/{}/similarity_networks/{}_full_mask_interactions_train_{}.gml".format(dataset, similarity_network, split))
            datasets['splits'][split][similarity_network]['val'] = nx.read_gml(
                "data/{}/similarity_networks/{}_full_mask_interactions_val_{}.gml".format(dataset, similarity_network, split))

            if similarity_network != "textSim_graph" and threshold_to_cut != 1.0:
                q = threshold_to_cut

                G = datasets['splits'][split][similarity_network]['train'].copy()
                weights = [e[2] for e in G.edges.data('weight')]
                threshold = np.quantile(np.array(weights), q)

                long_edges = list(filter(lambda e: e[2] > threshold, (e for e in G.edges.data('weight'))))
                le_ids = list(e[:2] for e in long_edges)

                G.remove_edges_from(le_ids)
                isolated_nodes = list(nx.isolates(G))
                G.remove_nodes_from(isolated_nodes)
                datasets['splits'][split][similarity_network]['train'] = G.copy()

                G = datasets['splits'][split][similarity_network]['val'].copy()
                weights = [e[2] for e in G.edges.data('weight')]
                threshold = np.quantile(np.array(weights), q)

                long_edges = list(filter(lambda e: e[2] > threshold, (e for e in G.edges.data('weight'))))
                le_ids = list(e[:2] for e in long_edges)

                G.remove_edges_from(le_ids)
                isolated_nodes = list(nx.isolates(G))
                G.remove_nodes_from(isolated_nodes)
                datasets['splits'][split][similarity_network]['val'] = G.copy()

    path_output = 'data/processed/{}/datasets_full_{}.pkl'.format(dataset, threshold_to_cut)

    with open(path_output, 'wb') as handle:
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    for similarity_network in similarity_networks:
        print(similarity_network)
        datasets[similarity_network] = {'train': None, 'val': None}

        if similarity_network == "final_graph":
            datasets[similarity_network]['train'] = mergeNetworks(
                [datasets["coRetweet_graph"]['train'],
                 datasets["coURL_graph"]['train'],
                 datasets["textSim_graph"]['train']])
            nx.write_gml(datasets['final_graph']['train'],
                         "data/{}/similarity_networks/final_graph_full_temporal_train.gml".format(
                             dataset))

            datasets[similarity_network]['val'] = mergeNetworks(
                [datasets["coRetweet_graph"]['val'],
                 datasets["coURL_graph"]['val'],
                 datasets["textSim_graph"]['val']])
            nx.write_gml(datasets['final_graph']['val'],
                         "data/{}/similarity_networks/final_graph_full_temporal_val.gml".format(
                             dataset))
        datasets[similarity_network]['train'] = nx.read_gml(
            "data/{}/similarity_networks/{}_full_temporal_train.gml".format(dataset,
                                                                                        similarity_network))
        datasets[similarity_network]['val'] = nx.read_gml(
            "data/{}/similarity_networks/{}_full_temporal_val.gml".format(dataset,
                                                                                          similarity_network))
        if similarity_network != "textSim_graph" and threshold_to_cut != 1.0:
            q = threshold_to_cut

            G = datasets[similarity_network]['train'].copy()
            weights = [e[2] for e in G.edges.data('weight')]
            threshold = np.quantile(np.array(weights), q)

            long_edges = list(filter(lambda e: e[2] > threshold, (e for e in G.edges.data('weight'))))
            le_ids = list(e[:2] for e in long_edges)

            G.remove_edges_from(le_ids)
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)
            datasets[similarity_network]['train'] = G.copy()

            G = datasets[similarity_network]['val'].copy()
            weights = [e[2] for e in G.edges.data('weight')]
            threshold = np.quantile(np.array(weights), q)

            long_edges = list(filter(lambda e: e[2] > threshold, (e for e in G.edges.data('weight'))))
            le_ids = list(e[:2] for e in long_edges)

            G.remove_edges_from(le_ids)
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)
            datasets[similarity_network]['val'] = G.copy()

    path_output = 'data/processed/{}/datasets_full_temporal_{}.pkl'.format(dataset, threshold_to_cut)

    with open(path_output, 'wb') as handle:
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

