import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

dataset = sys.argv[1]

df = pd.read_pickle("data/{}/interactions_and_tweets_graph_full.pkl".format(dataset))

user_hashtags = {}
for g in df[['userid', 'hashtags']].dropna().groupby('userid'):
    user_hashtags[g[0]] = np.unique(np.concatenate(g[1]['hashtags'].to_numpy()))

io_hashtags = {}
for g in df[df['group'] == "IO"][['userid', 'hashtags']].dropna().groupby('userid'):
    io_hashtags[g[0]] = np.unique(np.concatenate(g[1]['hashtags'].to_numpy()))

io_hashtags = np.unique(np.concatenate(list(io_hashtags.values())))

control_hashtags = {}
for g in df[df['group'] == "Control"][['userid', 'hashtags']].dropna().groupby('userid'):
    control_hashtags[g[0]] = np.unique(np.concatenate(g[1]['hashtags'].to_numpy()))

control_hashtags = np.unique(np.concatenate(list(control_hashtags.values())))

intersec = set(io_hashtags).intersection(set(control_hashtags))

import re

hashtags_to_take = list(intersec)
hashtags_to_take = list(set([re.sub(r'[_\W]+', '', x) for x in hashtags_to_take]))
hashtags_to_take = [x for x in hashtags_to_take if len(x) > 0]
hashtags_to_take = list(set([x.lower() for x in hashtags_to_take]))

if dataset == "cuba":
    hashtags_to_take.remove('cuba')
hashtags_to_take = sorted(hashtags_to_take)


def clean_list_hashtags(lst):
    if type(lst) != list and pd.isna(lst):
        return []
    return [re.sub(r'[_\W]+', '', x).lower() for x in lst]


df['clean_hashtags'] = df['hashtags'].apply(clean_list_hashtags)

control_users = df[df['group'] == "Control"]['userid'].unique()

df_exploded = df.explode('clean_hashtags')

exposed_adopted_users = []

i = 0

for hashtag in tqdm(hashtags_to_take):

    # --- FAST HASHTAG FILTER
    mask_hashtag = df_exploded['clean_hashtags'] == hashtag
    df_h = df_exploded[mask_hashtag]

    # Precompute shared masks
    mask_control = df_h['group'] == "Control"
    mask_not_control_io = df_h['interaction_type'].isin(["Control->Control", "Unknown"])# != "Control->IO"
    mask_io_control = df_h['interaction_type'] == "IO->Control"
    mask_control_control = df_h['interaction_type'] == "Control->Control"
    mask_not_retweet_only = df_h['interaction_form'] != "retweet_only"

    # --- Control users exposed for this hashtag
    control_users_exposed = df_h[mask_control]['userid'].unique()

    # --- Adopted (Control, non-Control->IO)
    df_adopted = (
        df_h[mask_control & mask_not_control_io]
        [['userid', 'interaction_with_userid', 'time']]
        .groupby('userid')
    )

    # --- Exposed IO->Control
    df_exposed_io = (
        df_h[
            df_h['interaction_with_userid'].isin(control_users_exposed)
            & mask_io_control
            & mask_not_retweet_only
            ][['userid', 'interaction_with_userid', 'time']]
        .groupby('interaction_with_userid')
    )

    # --- Exposed Control->Control
    df_exposed_cc = (
        df_h[
            df_h['interaction_with_userid'].isin(control_users_exposed)
            & mask_control_control
            & mask_not_retweet_only
            ][['userid', 'interaction_with_userid', 'time']]
        .groupby('interaction_with_userid')
    )

    # ===== MAIN LOOP OVER CONTROL USERS =====
    for c_user in control_users:

        exposure_io_count = 0
        first_io_exposure = -1
        last_io_exposure = -1
        # ---- IO→Control Exposures
        if c_user in df_exposed_io.groups:
            g = df_exposed_io.get_group(c_user)
            exposure_io_count = g.count().iloc[0]
            first_io_exposure = g['time'].min()
            last_io_exposure = g['time'].max()

        exposure_cc_count = 0
        first_cc_exposure = -1
        last_cc_exposure = -1
        # ---- Control→Control Exposures
        if c_user in df_exposed_cc.groups:
            g = df_exposed_cc.get_group(c_user)
            exposure_cc_count = g.count().iloc[0]
            first_cc_exposure = g['time'].min()
            last_cc_exposure = g['time'].max()

        adopted_count_before_first_io_exposure = 0
        adopted_count_before_last_io_exposure = 0

        adopted_count_after_first_io_exposure = 0
        adopted_count_after_last_io_exposure = 0

        adopted_count_before_first_cc_exposure = 0
        adopted_count_before_last_cc_exposure = 0

        adopted_count_after_first_cc_exposure = 0
        adopted_count_after_last_cc_exposure = 0

        if c_user in df_adopted.groups:
            adopted_group = df_adopted.get_group(c_user)

            if first_io_exposure != -1:
                adopted_count_before_first_io_exposure = (adopted_group['time'] < first_io_exposure).sum()
                adopted_count_after_first_io_exposure = (adopted_group['time'] > first_io_exposure).sum()

            if last_io_exposure != -1:
                adopted_count_before_last_io_exposure = (adopted_group['time'] < last_io_exposure).sum()
                adopted_count_after_last_io_exposure = (adopted_group['time'] > last_io_exposure).sum()

            if first_cc_exposure != -1:
                adopted_count_before_first_cc_exposure = (adopted_group['time'] < first_cc_exposure).sum()
                adopted_count_after_first_cc_exposure = (adopted_group['time'] > first_cc_exposure).sum()

            if last_cc_exposure != -1:
                adopted_count_before_last_cc_exposure = (adopted_group['time'] < last_cc_exposure).sum()
                adopted_count_after_last_cc_exposure = (adopted_group['time'] > last_cc_exposure).sum()

        if exposure_io_count + exposure_cc_count + adopted_count_before_last_io_exposure + adopted_count_after_last_io_exposure +\
            adopted_count_before_last_cc_exposure + adopted_count_after_last_cc_exposure > 0:
            exposed_adopted_users.append((c_user, hashtag, exposure_io_count, exposure_cc_count,
                                                               adopted_count_before_first_io_exposure, adopted_count_after_first_io_exposure,
                                                               adopted_count_before_last_io_exposure, adopted_count_after_last_io_exposure,
                                                               adopted_count_before_first_cc_exposure, adopted_count_after_first_cc_exposure,
                                                               adopted_count_before_last_cc_exposure, adopted_count_after_last_cc_exposure))

if len(exposed_adopted_users) > 100000:
    exposed_adopted_users_df = pd.DataFrame(exposed_adopted_users,
                                            columns=["userid", "hashtag", "NumExposureFromIO",
                                                     "NumExposureFromControl",
                                                     "NumAdoptionsBeforeFirstIOExposure",
                                                     "NumAdoptionsAfterFirstIOExposure",
                                                     "NumAdoptionsBeforeLastIOExposure",
                                                     "NumAdoptionsAfterLastIOExposure",
                                                     "NumAdoptionsBeforeFirstCCExposure",
                                                     "NumAdoptionsAfterFirstCCExposure",
                                                     "NumAdoptionsBeforeLastCCExposure",
                                                     "NumAdoptionsAfterLastCCExposure",
                                                     ])

    if full:
        exposed_adopted_users_df.to_csv("data/{}/hashtags_exposed_adopted_users_full_{}.csv.gz".format(dataset, i),
                                        index=False, compression="gzip")
    else:
        exposed_adopted_users_df.to_csv("data/{}/hashtags_exposed_adopted_users_{}.csv.gz".format(dataset, i),
                                        index=False, compression="gzip")
    print("Saved", i)
    i += 1

    del exposed_adopted_users
    exposed_adopted_users = []

exposed_adopted_users_df = pd.DataFrame(exposed_adopted_users,
                                        columns=["userid", "hashtag", "NumExposureFromIO",
                                                 "NumExposureFromControl",
                                                 "NumAdoptionsBeforeFirstIOExposure",
                                                 "NumAdoptionsAfterFirstIOExposure",
                                                 "NumAdoptionsBeforeLastIOExposure",
                                                 "NumAdoptionsAfterLastIOExposure",
                                                 "NumAdoptionsBeforeFirstCCExposure",
                                                 "NumAdoptionsAfterFirstCCExposure",
                                                 "NumAdoptionsBeforeLastCCExposure",
                                                 "NumAdoptionsAfterLastCCExposure",
                                                 ])

exposed_adopted_users_df.to_csv("data/{}/hashtags_exposed_adopted_users_full_{}.csv.gz".format(dataset, i),
                                index=False, compression="gzip")
print("Saved", i)

