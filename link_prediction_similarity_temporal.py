import os.path
import sys
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pickle
import networkx as nx
from utils import mergeNetworks

dataset = sys.argv[1]

with open('data/processed/{}/datasets_full_temporal_1.0.pkl'.format(dataset), 'rb') as f:
    datasets = pickle.load(f)


similarity_networks = ["coRetweet_graph", "coURL_graph", "textSim_graph", "final_graph"]

for similarity_network in similarity_networks:
    test_ap = []
    test_auc = []
    graph = datasets[similarity_network]['val']
    for split in datasets['splits']:

        pos_edges = datasets['splits'][split]['test_positive_edges']
        neg_edges = datasets['splits'][split]['test_negative_edges']

        pos_prob = []
        for edge in pos_edges:
            try:
                pos_prob.append(graph.edges[edge]['weight'])
            except:
                pos_prob.append(0.0)

        neg_prob = []
        for edge in neg_edges:
            try:
                neg_prob.append(graph.edges[edge]['weight'])
            except:
                neg_prob.append(0.0)

        pred_score = np.concatenate([pos_prob, neg_prob])
        true_label = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])

        test_ap.append(average_precision_score(true_label, pred_score))
        test_auc.append(roc_auc_score(true_label, pred_score))

        print(f"Similarity {similarity_network} Trial {split}; AUC: {test_auc[-1]}, AP: {test_ap[-1]}")

    auc_mean = np.around(np.mean(test_auc), 3)
    auc_std = np.around(np.std(test_auc), 3)

    ap_mean = np.around(np.mean(test_ap), 3)
    ap_std = np.around(np.std(test_ap), 3)

    print(f"Similarity {similarity_network} AUC: {auc_mean} +- {auc_std}, AP: {ap_mean} +- {ap_std}")

print("Finish", dataset)
