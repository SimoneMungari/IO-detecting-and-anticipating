import pickle
import pandas as pd
import gzip
import networkx as nx

def load_data(dataset):
    if dataset == "china":
        with open('data/{}/{}_082019_1_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pickle.load(f)

        with open('data/{}/{}_082019_1_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pickle.load(f)

        with open('data/{}/{}_082019_2_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pd.concat((io_df, pickle.load(f)))

        with open('data/{}/{}_082019_2_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pd.concat((control_df, pickle.load(f)))

    elif dataset == "cuba":
        with open('data/{}/{}_082020_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pickle.load(f)

        with open('data/{}/{}_082020_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pickle.load(f)

    elif dataset == "russia":
        with open('data/{}/{}_201901_1_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pickle.load(f)

        with open('data/{}/{}_201901_1_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pickle.load(f)

    elif dataset == "uae":
        with open('data/{}/{}_082019_1_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pickle.load(f)

        with open('data/{}/{}_082019_1_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pickle.load(f)

    elif dataset == "iran":

        with open('data/{}/{}_201901_1_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pickle.load(f)

        with open('data/{}/{}_201906_2_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pd.concat((io_df, pickle.load(f)))

        with open('data/{}/{}_201906_3_tweets_io.pkl'.format(dataset, dataset), 'rb') as f:
            io_df = pd.concat((io_df, pickle.load(f)))

        with open('data/{}/{}_201906_1_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pickle.load(f)

        with open('data/{}/{}_201906_2_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pd.concat((control_df, pickle.load(f)))

        with open('data/{}/{}_201906_3_tweets_control.pkl'.format(dataset, dataset), 'rb') as f:
            control_df = pd.concat((control_df, pickle.load(f)))

    elif dataset == "qatar":
        with gzip.open('data/{}/{}_082020_tweets_io.pkl.gz'.format(dataset, dataset), 'rb') as f:
            io_df = pickle.load(f)

        with gzip.open('data/{}/{}_082020_tweets_control.pkl.gz'.format(dataset, dataset), 'rb') as f:
            control_df = pickle.load(f)

    return io_df, control_df


def mergeNetworks(singleFeatureNets, weighted=True):
    """
    Merges multiple networks in a single one, where two nodes will be connected if they are in at least one of the input networks.

    Args:
        singleFeatureNets: List of networks to merge, each network must be a networkx.Graph object
        weighted: boolean variable indicating if the merged network should me weighted or not. If True, multiple weights for the same edge are grouped taking the maximum.
    Returns:
        M: merged network
    """

    graphs = []

    for net in singleFeatureNets:
        if weighted:
            df = pd.DataFrame(net.edges(data='weight'))
        else:
            df = pd.DataFrame(net.edges())
        graphs.append(df)

    temp = pd.concat([df for df in graphs])
    temp = temp.loc[temp[0] != temp[1]]

    if weighted:
        temp.columns = ['source', 'target', 'weight']
        temp = temp.groupby(['source', 'target'], as_index=False).max()
    else:
        temp.columns = ['source', 'target']

    temp.dropna(inplace=True)

    if weighted:
        M = nx.from_pandas_edgelist(temp, edge_attr=True)
    else:
        M = nx.from_pandas_edgelist(temp)

    return M