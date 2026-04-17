import os.path

import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import gc
import numpy as np
import torch
from os import listdir
import warnings
import datetime
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import timedelta
from text_preprocessing import *
import sys

from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
import pickle

from utils import load_data, mergeNetworks

def check_memory_usage():
    import os
    import psutil

    process = psutil.Process(os.getpid())
    # Get memory info as a named tuple
    mem_info = process.memory_info()

    # Print the RSS (Resident Set Size) in MiB (Megabytes)
    print(f"Current process memory usage: {mem_info.rss / (1024 ** 2):.2f} MiB")

def coRetweet(control, treated):
    control.dropna(inplace=True)
    control = control[control['retweet_tweetid'] != 'None']
    treated.dropna(inplace=True)

    control['userid'] = control['userid'].astype(int)
    control['retweet_tweetid'] = control['retweet_tweetid'].astype(int)

    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)

    cum = pd.concat([treated, control])
    filt = cum[['userid', 'tweetid']].groupby(['userid'], as_index=False).count()
    filt = list(filt.loc[filt['tweetid'] >= 20]['userid'])
    cum = cum.loc[cum['userid'].isin(filt)]
    cum = cum[['userid', 'retweet_tweetid']].drop_duplicates()

    temp = cum.groupby('retweet_tweetid', as_index=False).count()
    cum = cum.loc[cum['retweet_tweetid'].isin(temp.loc[temp['userid'] > 1]['retweet_tweetid'].to_list())]

    cum['value'] = 1

    ids = dict(zip(list(cum.retweet_tweetid.unique()), list(range(cum.retweet_tweetid.unique().shape[0]))))
    cum['retweet_tweetid'] = cum['retweet_tweetid'].apply(lambda x: ids[x]).astype(int)
    del ids

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_tweetid.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_tweetid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def coURL(control, treated):
    control.dropna(inplace=True)

    control = control[['userid', 'urls']].explode('urls')
    control.dropna(inplace=True)

    treated['urls'] = treated['urls'].astype(str).replace('[]', '').apply(
        lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    treated = treated.loc[treated['urls'] != ''].explode('urls')

    cum = pd.concat([control, treated])[['userid', 'urls']].dropna()
    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('urls', as_index=False).count()
    cum = cum.loc[cum['urls'].isin(temp.loc[temp['userid'] > 1]['urls'].to_list())]

    cum['value'] = 1
    urls = dict(zip(list(cum.urls.unique()), list(range(cum.urls.unique().shape[0]))))
    cum['urls'] = cum['urls'].apply(lambda x: urls[x]).astype(int)
    del urls

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.urls.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.urls.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
        return utcdttime
    except:
        return None

def fastRetweet(control, treated, timeInterval=10):
    control.dropna(inplace=True)
    control = control[control['retweet_tweetid'] != 'None']
    treated.dropna(inplace=True)

    control['retweet_tweetid'] = control['retweet_tweetid'].astype(int)
    control['retweet_userid'] = control['retweet_userid'].astype(int)
    control['userid'] = control['userid'].astype(int)
    control['tweet_timestamp'] = control['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    control['retweet_timestamp'] = control['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)
    treated['tweet_timestamp'] = treated['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    treated['retweet_timestamp'] = treated['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    treated['delta'] = (treated['tweet_timestamp'] - treated['retweet_timestamp']).dt.seconds
    control['delta'] = (control['tweet_timestamp'] - control['retweet_timestamp']).dt.seconds

    cumulative = pd.concat(
        [treated[['userid', 'retweet_userid', 'delta']], control[['userid', 'retweet_userid', 'delta']]])
    cumulative['userid'] = cumulative['userid'].astype(int).astype(str)
    cumulative = cumulative.loc[cumulative['delta'] <= timeInterval]

    cumulative = cumulative.groupby(['userid', 'retweet_userid'], as_index=False).count()
    cum = cumulative.loc[cumulative['delta'] > 1]

    urls = dict(zip(list(cum.retweet_userid.unique()), list(range(cum.retweet_userid.unique().shape[0]))))
    cum['retweet_userid'] = cum['retweet_userid'].apply(lambda x: urls[x]).astype(int)
    del urls

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_userid.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_userid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["delta"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def hashSeq(control, treated, minHashtags=5):
    control.replace(np.NaN, None, inplace=True)
    control.replace('None', None, inplace=True)

    control['engagementParentId'] = control['in_reply_to_tweetid']

    retweet_id = []
    names = []
    eng = []
    for row in control[['retweet_tweetid', 'userid', 'in_reply_to_tweetid']].values:
        if row[0] != None:
            u = row[0]
            retweet_id.append(u)
            eng.append('retweet')
        elif row[2] != None:
            retweet_id.append(row[2])
            eng.append('reply')
        else:
            retweet_id.append(None)
            eng.append('tweet')
        u = row[1]
        names.append(u)

    control['twitterAuthorScreenname'] = names
    control['retweet_ordinalId'] = retweet_id
    control['engagementType'] = eng
    control['engagementParentId'].fillna(control['retweet_ordinalId'], inplace=True)

    control_filt = control[['twitterAuthorScreenname', 'engagementType', 'engagementParentId']]
    control_filt['contentText'] = control['tweet_text']
    control_filt['tweetId'] = control['tweetid'].astype(int)
    control_filt['tweet_timestamp'] = control_filt['tweetId'].apply(lambda x: get_tweet_timestamp(x))

    del control

    treated.replace(np.NaN, None, inplace=True)
    treated.replace('None', None, inplace=True)

    retweet_id = []
    names = []
    eng = []
    for row in treated[['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid']].values:
        if row[0] != None:
            retweet_id.append(row[0])
            eng.append('retweet')
        elif row[2] != None:
            retweet_id.append(row[2])
            eng.append('reply')
        elif row[3] != None:
            retweet_id.append(row[3])
            eng.append('quote tweet')
        else:
            retweet_id.append(None)
            eng.append('tweet')
        names.append(row[1])

    treated['twitterAuthorScreenname'] = names
    treated['engagementType'] = eng
    treated['engagementParentId'] = retweet_id

    treated_filt = treated[['twitterAuthorScreenname', 'engagementType', 'engagementParentId']]
    treated_filt['contentText'] = treated['tweet_text']
    treated_filt['tweetId'] = treated['tweetid'].astype(int)
    treated_filt['tweet_timestamp'] = treated_filt['tweetId'].apply(lambda x: get_tweet_timestamp(x))

    del treated

    cum = pd.concat([control_filt, treated_filt])

    del control_filt, treated_filt

    cum = cum.loc[cum['engagementType'] != 'retweet']
    cum['hashtag_seq'] = ['__'.join([tag.strip("#") for tag in tweet.split() if tag.startswith("#")]) for tweet in
                          cum['contentText'].values.astype(str)]
    cum.drop('contentText', axis=1, inplace=True)
    cum = cum[['twitterAuthorScreenname', 'hashtag_seq']].loc[
        cum['hashtag_seq'].apply(lambda x: len(x.split('__'))) >= 1]

    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('hashtag_seq', as_index=False).count()
    cum = cum.loc[cum['hashtag_seq'].isin(temp.loc[temp['twitterAuthorScreenname'] > minHashtags]['hashtag_seq'].to_list())]

    cum['value'] = 1

    hashs = dict(zip(list(cum.hashtag_seq.unique()), list(range(cum.hashtag_seq.unique().shape[0]))))
    cum['hashtag_seq'] = cum['hashtag_seq'].apply(lambda x: hashs[x]).astype(int)
    del hashs

    userid = dict(zip(list(cum.twitterAuthorScreenname.astype(str).unique()),
                      list(range(cum.twitterAuthorScreenname.unique().shape[0]))))
    cum['twitterAuthorScreenname'] = cum['twitterAuthorScreenname'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.twitterAuthorScreenname.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.hashtag_seq.unique()), ordered=True)

    row = cum.twitterAuthorScreenname.astype(person_c).cat.codes
    col = cum.hashtag_seq.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)
    print("similarities computed")

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    print("Before networkx")
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def textSim(control, treated, outputDir, timeWindow=365, time_span=30):

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    control['tweetid'] = control['tweetid'].astype(int)
    treated['tweetid'] = treated['tweetid'].astype(int)

    treated = get_positive_data(treated)
    control['tweet_time'] = control['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    pos_en_df_all = preprocess_text(treated)
    del treated
    neg_en_df_all = preprocess_text(control)
    del control

    pos_en_df_all['tweet_text'] = pos_en_df_all['tweet_text'].replace(',', '')
    neg_en_df_all['tweet_text'] = neg_en_df_all['tweet_text'].replace(',', '')

    pos_en_df_all['clean_tweet'] = pos_en_df_all['tweet_text'].astype(str).apply(lambda x: msg_clean(x))
    neg_en_df_all['clean_tweet'] = neg_en_df_all['tweet_text'].astype(str).apply(lambda x: msg_clean(x))

    pos_en_df_all = pos_en_df_all[pos_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    neg_en_df_all = neg_en_df_all[neg_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    pos_en_df_all['tweet_time'] = pos_en_df_all['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    if neg_en_df_all.shape[0] > 0:
        neg_en_df_all['tweet_time'] = neg_en_df_all['tweetid'].apply(lambda x: get_tweet_timestamp(x))

    date = pos_en_df_all['tweet_time'].min().date()
    finalDate = pos_en_df_all['tweet_time'].max().date()

    def create_sim_score_df(lims, D, I, search_query1):
        source_idx = []
        target_idx = []
        sim_score = []

        for i in range(len(search_query1)):
            idx = I[lims[i]:lims[i + 1]]
            sim = D[lims[i]:lims[i + 1]]

            # print(idx.shape, sim.shape)
            for j in range(len(idx)):
                source_idx.append(i)
                target_idx.append(idx[j])
                sim_score.append(sim[j])

        sim_score_df = pd.DataFrame(list(zip(source_idx, target_idx, sim_score)),
                                    columns=['source_idx', 'target_idx', 'sim_score'])

        # print("sim score df")
        # check_memory_usage()

        del source_idx
        del target_idx
        del sim_score
        sim_score_df = sim_score_df.query("source_idx != target_idx")
        sim_score_df['combined_idx'] = sim_score_df[['source_idx', 'target_idx']].apply(tuple, axis=1)
        sim_score_df['combined_idx'] = sim_score_df['combined_idx'].apply(sorted)
        sim_score_df['combined_idx'] = sim_score_df['combined_idx'].transform(lambda k: tuple(k))
        sim_score_df = sim_score_df.drop_duplicates(subset=['combined_idx'], keep='first')
        sim_score_df.reset_index(inplace=True)
        sim_score_df = sim_score_df.loc[:, ~sim_score_df.columns.str.contains('index')]
        sim_score_df.drop(['combined_idx'], inplace=True, axis=1)

        df_join = pd.merge(
            pd.merge(sim_score_df, combined_tweets_df, left_on='source_idx', right_on='my_idx', how='inner'),
            combined_tweets_df, left_on='target_idx', right_on='my_idx', how='inner')

        # print("df join")
        # check_memory_usage()

        result = df_join[['userid_x', 'userid_y', 'clean_tweet_x', 'clean_tweet_y', 'sim_score']]
        result = result.rename(columns={'userid_x': 'source_user',
                                        'userid_y': 'target_user',
                                        'clean_tweet_x': 'source_text',
                                        'clean_tweet_y': 'target_text'})

        del df_join
        del sim_score_df

        return result

    i = 1

    print(finalDate)
    while date <= finalDate:
        print("start")
        check_memory_usage()
        # threshold = 0.7
        #
        if os.path.exists(outputDir + '/threshold_0.7_' + str(i) + '.csv'):
            print(date, "already done")
            date = date + timedelta(days=time_span)
            i += 1
            continue

        import sys
        sys.stdout.flush()

        pos_en_df = pos_en_df_all.loc[(pos_en_df_all['tweet_time'].dt.date >= date) & (
                    pos_en_df_all['tweet_time'].dt.date < date + timedelta(days=timeWindow))]

        if neg_en_df_all.shape[0] > 0:
            neg_en_df = neg_en_df_all.loc[(neg_en_df_all['tweet_time'].dt.date >= date) & (
                        neg_en_df_all['tweet_time'].dt.date < date + timedelta(days=timeWindow))]

            combined_tweets_df = pd.concat([pos_en_df, neg_en_df], axis=0)
        else:
            combined_tweets_df = pos_en_df

        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.loc[:, ~combined_tweets_df.columns.str.contains('index')]
        print(combined_tweets_df.shape)

        if len(combined_tweets_df) > 50000:
            idxs = np.random.choice(np.arange(len(combined_tweets_df)), replace=False, size=50000)
            combined_tweets_df = combined_tweets_df.iloc[idxs]

        # print(combined_tweets_df.shape)

        del pos_en_df
        if neg_en_df_all.shape[0] > 0:
            del neg_en_df

        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.rename(columns={'index': 'my_idx'})

        sentences = combined_tweets_df.clean_tweet.tolist()
        # check_memory_usage()

        encoder = SentenceTransformer('stsb-xlm-r-multilingual').to('cuda:0')
        # print(encoder.device)
        # print("before encoding")
        plot_embeddings = encoder.encode(sentences)

        try:
            dim = plot_embeddings.shape[1]  # vector dimension
        except:
            del combined_tweets_df
            del sentences
            gc.collect()

            print(date, "no embeddings")
            date = date + timedelta(days=time_span)

            continue

        # print("Before dbvectors1")
        # check_memory_usage()

        db_vectors1 = plot_embeddings.copy().astype(np.float32)
        a = [j for j in range(plot_embeddings.shape[0])]
        db_ids1 = np.array(a, dtype=np.int64)

        # print("Before normalize")
        # check_memory_usage()
        faiss.normalize_L2(db_vectors1)

        # print("Before index")
        # check_memory_usage()
        index1 = faiss.IndexFlatIP(dim)
        index1 = faiss.IndexIDMap(index1)  # mapping df index as id
        index1.add_with_ids(db_vectors1, db_ids1)

        search_query1 = plot_embeddings.copy().astype(np.float32)

        plot_embeddings = None
        del plot_embeddings
        del db_vectors1
        del a
        del db_ids1
        del encoder

        gc.collect()
        torch.cuda.empty_cache()

        # print("Before normalize 2")
        # check_memory_usage()

        faiss.normalize_L2(search_query1)

        result_plot_thres = []
        result_plot_score = []
        result_plot_metrics = []

        init_threshold = 0.7

        # print("Before retrieved")
        # check_memory_usage()
        lims, D, I = index1.range_search(x=search_query1, thresh=init_threshold)
        # print('Retrieved results of index search')
        # check_memory_usage()

        sim_score_df = create_sim_score_df(lims, D, I, search_query1)
        # print('Generated Similarity Score DataFrame')

        del combined_tweets_df
        del lims
        del D
        del I
        del search_query1
        index1.reset()
        del index1


        # print("before threshold")
        # check_memory_usage()

        for threshold in np.arange(0.7, 1.01, 0.05):
            # print("Threshold: ", threshold)

            sim_score_temp_df = sim_score_df[
                (sim_score_df.sim_score >= threshold) & (sim_score_df.sim_score < threshold + 0.05)]

            text_sim_network = sim_score_temp_df[['source_user', 'target_user']]

            del sim_score_temp_df

            text_sim_network = pd.DataFrame(text_sim_network.value_counts(subset=(['source_user', 'target_user'])))
            text_sim_network.reset_index(inplace=True)
            text_sim_network.columns = ['source_user', 'target_user', 'count']

            # print("before saving")
            # check_memory_usage()
            outputfile = outputDir + '/threshold_' + str(threshold) + '_' + str(i) + '.csv'
            text_sim_network.to_csv(outputfile)

            del text_sim_network

        del sim_score_df

        print(date, "finished")
        date = date + timedelta(days=time_span)
        i += 1
        gc.collect()


def getSimilarityNetwork(inputDir):

    if not os.path.exists(inputDir):
        os.makedirs(inputDir)

    files = [f for f in listdir(inputDir)]
    files.sort()

    d = {'threshold_1.00': [],
         'threshold_0.90': [],
         'threshold_0.95': [],
         'threshold_0.85': [],
         'threshold_0.8': [],
         'threshold_0.75': [],
         'threshold_0.7': []}

    for f in files:
        if f[:9] == 'threshold':
            d['_'.join(f[:-4].split('_')[:2])[:14]].append(f)

    path = inputDir
    i = 0

    for fil in d.keys():
        thr = float(fil.split('_')[-1][:4])

        l = d[fil]
        if i == 0:
            combined = pd.read_csv(os.path.join(path, l[0]))
            combined['weight'] = thr
            i += 1
            for o in l[1:]:
                temp = pd.read_csv(os.path.join(path, o))
                temp['weight'] = thr
                combined = pd.concat([combined, temp])
        else:
            for o in l:
                temp = pd.read_csv(os.path.join(path, o))
                temp['weight'] = thr
                combined = pd.concat([combined, temp])

        # if i == 3:
        #     break

    combined = combined.groupby(['source_user', 'target_user', 'weight'], as_index=False).sum()
    combined['weight'] = combined['weight'] * combined['count']
    combined = combined.groupby(['source_user', 'target_user'], as_index=False).sum()
    combined['weight'] = combined['weight'] / combined['count']

    G = nx.from_pandas_edgelist(combined[['source_user', 'target_user', 'weight']], source='source_user',
                                target='target_user', edge_attr=['weight'])

    return G

if __name__ == '__main__':

    dataset = sys.argv[1]
    mask_interactions = True if len(sys.argv) > 2 and sys.argv[2] == "1" else False

    print(f"Dataset {dataset}, Mask Interactions {mask_interactions}")

    io_df, control_df = load_data(dataset)

    if not os.path.exists("data/{}/similarity_networks".format(dataset)):
        os.makedirs("data/{}/similarity_networks".format(dataset))


    io_df = io_df.groupby('userid').filter(lambda x: len(x) >= 10)
    control_df = control_df.groupby('userid').filter(lambda x: len(x) >= 10)

    io_df['tweet_time'] = pd.to_datetime(pd.to_datetime(io_df['tweet_time'], format="mixed"), utc=True).dt.tz_localize(
        None)
    control_df['tweet_time'] = pd.to_datetime(pd.to_datetime(control_df['tweet_time'], format="mixed"),
                                              utc=True).dt.tz_localize(None)
    max_date = max(io_df['tweet_time'].max(), control_df['tweet_time'].max())

    if mask_interactions:
        with open('data/processed/{}/datasets_full.pkl'.format(dataset), 'rb') as f:
            datasets = pickle.load(f)

        df_interactions = pd.read_pickle("data/{}/interactions_and_tweets_graph_full.pkl".format(dataset))

    output_path = "graph"
    output_path += "_full"

    if mask_interactions:
        output_path += "_mask_interactions"
        for split in [0, 1, 2, 3, 4]:
            output_path_train = output_path + "_train_{}.gml".format(split)
            output_path_val = output_path + "_val_{}.gml".format(split)

            edges_to_remove = np.concatenate(
                (datasets['splits'][split]['val_positive_edges'], datasets['splits'][split]['test_positive_edges']))
            edges_to_remove = list(map(tuple, edges_to_remove))
            df_interactions['edge'] = list(map(tuple, df_interactions[['userid', 'interaction_with_userid']].to_numpy()))
            df_interactions_train = df_interactions[~df_interactions['edge'].isin(edges_to_remove)]

            io_df_train = io_df.copy()
            io_df_train = io_df_train[io_df_train['tweetid'].isin(df_interactions_train['tweetid'].unique())]
            control_df_train = control_df.copy()
            control_df_train = control_df_train[control_df_train['tweetid'].isin(df_interactions_train['tweetid'].unique())]

            edges_to_remove = datasets['splits'][split]['test_positive_edges']
            edges_to_remove = list(map(tuple, edges_to_remove))
            df_interactions_val = df_interactions[~df_interactions['edge'].isin(edges_to_remove)]

            io_df_val = io_df.copy()
            io_df_val = io_df_val[io_df_val['tweetid'].isin(df_interactions_val['tweetid'].unique())]
            control_df_val = control_df.copy()
            control_df_val = control_df_val[control_df_val['tweetid'].isin(df_interactions_val['tweetid'].unique())]

            coRetweet_graph_train = coRetweet(control_df_train[['userid', 'retweet_tweetid', 'tweetid']],
                                        io_df_train[['tweetid', 'userid', 'retweet_tweetid']])
            nx.write_gml(coRetweet_graph_train, "data/{}/similarity_networks/coRetweet_{}".format(dataset, output_path_train))

            coRetweet_graph_val = coRetweet(control_df_val[['userid', 'retweet_tweetid', 'tweetid']],
                                              io_df_val[['tweetid', 'userid', 'retweet_tweetid']])
            nx.write_gml(coRetweet_graph_val,
                         "data/{}/similarity_networks/coRetweet_{}".format(dataset, output_path_val))

            print("CoRetweet saved")

            coURL_graph_train = coURL(control_df_train[['userid', 'tweetid', 'urls']], io_df_train[['userid', 'tweetid', 'urls']])
            nx.write_gml(coURL_graph_train, "data/{}/similarity_networks/coURL_{}".format(dataset, output_path_train))

            coURL_graph_val = coURL(control_df_val[['userid', 'tweetid', 'urls']],
                                      io_df_val[['userid', 'tweetid', 'urls']])
            nx.write_gml(coURL_graph_val, "data/{}/similarity_networks/coURL_{}".format(dataset, output_path_val))

            print("coURL saved")

            time_span = 30
            timeWindow = 30
            if dataset in ["russia", "iran", "qatar"]:
                timeWindow = 365

            if dataset == "china":
                time_span = 15
                timeWindow = 15

            output_textSim_train = "data/{}/similarity_networks/TextSimTrain_{}".format(dataset, split)
            output_textSim_train += "_full"

            textSim(control_df_train[['tweetid', 'tweet_text', 'tweet_language', 'tweet_time', 'userid']], io_df_train,
                    output_textSim_train, timeWindow, time_span)
            textSim_graph_train = getSimilarityNetwork(output_textSim_train)
            nx.write_gml(textSim_graph_train, "data/{}/similarity_networks/textSim_{}".format(dataset, output_path_train))

            output_textSim_val = "data/{}/similarity_networks/TextSimVal_{}".format(dataset, split)
            output_textSim_val += "_full"

            textSim(control_df_val[['tweetid', 'tweet_text', 'tweet_language', 'tweet_time', 'userid']], io_df_val,
                    output_textSim_val, timeWindow, time_span)
            textSim_graph_val = getSimilarityNetwork(output_textSim_val)
            nx.write_gml(textSim_graph_val, "data/{}/similarity_networks/textSim_{}".format(dataset, output_path_val))

            print("textSim saved")

    else:
        with open('data/processed/{}/datasets_full_temporal.pkl'.format(dataset), 'rb') as f:
            datasets = pickle.load(f)

        train_max_time = datasets['train_max_time']
        val_max_time = datasets['val_max_time']

        control_df_train = control_df[control_df['tweet_time'] < train_max_time]
        control_df_val = control_df[
            (control_df['tweet_time'] >= train_max_time) & (control_df['tweet_time'] < val_max_time)]

        io_df_train = io_df[io_df['tweet_time'] < train_max_time]
        io_df_val = io_df[(io_df['tweet_time'] >= train_max_time) & (io_df['tweet_time'] < val_max_time)]

        output_path_train = output_path + "_temporal_train.gml"
        output_path_val = output_path + "_temporal_val.gml"

        coRetweet_graph = coRetweet(control_df_train[['userid', 'retweet_tweetid', 'tweetid']], io_df_train[['tweetid', 'userid', 'retweet_tweetid']])
        nx.write_gml(coRetweet_graph, "data/{}/similarity_networks/coRetweet_{}".format(dataset, output_path_train))

        coRetweet_graph = coRetweet(control_df_val[['userid', 'retweet_tweetid', 'tweetid']],
                                    io_df_val[['tweetid', 'userid', 'retweet_tweetid']])
        nx.write_gml(coRetweet_graph, "data/{}/similarity_networks/coRetweet_{}".format(dataset, output_path_val))

        print("CoRetweet saved")

        coURL_graph = coURL(control_df_train[['userid', 'tweetid', 'urls']], io_df_train[['userid', 'tweetid', 'urls']])
        nx.write_gml(coURL_graph, "data/{}/similarity_networks/coURL_{}".format(dataset, output_path_train))

        coURL_graph = coURL(control_df_val[['userid', 'tweetid', 'urls']], io_df_val[['userid', 'tweetid', 'urls']])
        nx.write_gml(coURL_graph, "data/{}/similarity_networks/coURL_{}".format(dataset, output_path_val))

        print("coURL saved")

        time_span = 30
        timeWindow = 365

        if dataset == "china":
            time_span = 15
            timeWindow = 15

        output_textSim_train = "data/{}/similarity_networks/TextSimTrain_temporal".format(dataset)
        output_textSim_train += "_full"

        output_textSim_val = "data/{}/similarity_networks/TextSimVal_temporal".format(dataset)
        output_textSim_val += "_full"

        textSim(control_df_train[['tweetid', 'tweet_text', 'tweet_language', 'tweet_time', 'userid']], io_df_train,
               output_textSim_train, timeWindow=timeWindow, time_span=time_span)
        textSim_graph = getSimilarityNetwork(output_textSim_train)
        nx.write_gml(textSim_graph, "data/{}/similarity_networks/textSim_{}".format(dataset, output_path_train))

        textSim(control_df_val[['tweetid', 'tweet_text', 'tweet_language', 'tweet_time', 'userid']], io_df_val,
                output_textSim_val, timeWindow=timeWindow, time_span=time_span)
        textSim_graph = getSimilarityNetwork(output_textSim_val)
        nx.write_gml(textSim_graph, "data/{}/similarity_networks/textSim_{}".format(dataset, output_path_val))

        print("textSim saved")
