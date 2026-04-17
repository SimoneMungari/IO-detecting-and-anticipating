# IO-Impact-and-Prediction
Files to run for reproducing the auditing results:
1) python generate_interaction_network.py {dataset_name}
2) python analysis_over_time_hashtag.py {dataset_name}
3) python analysis_over_time_urls.py {dataset_name}

Files to run for reproducing the static link prediction results:
1) python generate_interaction_network.py {dataset_name}
2) python construct_dataset.py {dataset_name} 1
3) python generate_similarity_network.py {dataset_name} 1
4) python merge_similarity_networks.py {dataset_name} 0.3 0
5) python link_prediction_gnn.py {dataset_name} all
6) python link_prediction_gnn.py {dataset_name} interactions
7) python link_prediction_gnn.py {dataset_name} similarities
8) python link_prediction_node2vec_interaction.py {dataset_name}
9) python link_prediction_node2vec_similarity.py {dataset_name} {similarity_network} 0.3
10) python link_prediction_similarity.py {dataset_name}

Files to run for reproducing the temporal link prediction results:
1) python generate_interaction_network.py {dataset_name}
2) python construct_dataset.py {dataset_name} 0
3) python generate_similarity_network.py {dataset_name} 0
4) python merge_similarity_networks.py {dataset_name} 0.3 1
5) python link_prediction_gnn_temporal.py {dataset_name} all
6) python link_prediction_gnn_temporal.py {dataset_name} interactions
7) python link_prediction_gnn_temporal.py {dataset_name} similarities
8) python link_prediction_node2vec_interaction_temporal.py {dataset_name}
9) python link_prediction_node2vec_similarity_temporal.py {dataset_name} {similarity_network} 0.3
10) python link_prediction_tgnn_temporal.py {dataset_name} 
11) python link_prediction_tgnn_similarity_temporal.py {dataset_name}