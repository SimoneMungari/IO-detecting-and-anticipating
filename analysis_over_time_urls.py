import os.path
import pandas as pd
import numpy as np
import sys

dataset = sys.argv[1]

df = pd.read_pickle("data/{}/interactions_and_tweets_graph_full.pkl".format(dataset))
df = df[~pd.isna(df['urls'])]
df = df[~(df['urls'] == 'nan')]

print(df.columns)

users_urls = df[['tweetid', 'urls', 'group']].to_numpy()

io_urls = []
control_urls = []

clean_urls = []

io_urls_map = {}
control_urls_map = {}
for group in users_urls:
    if type(group[1]) != list:
        urls = eval(group[1])
    else:
        urls = group[1]

    if len(urls) > 0:

        urls_new = []
        for x in urls:
            try:
                urls_new.append(x.split("/")[2])
            except:
                continue
        urls = urls_new


        if group[2] == "IO":
            io_urls.extend(urls)
        else:
            control_urls.extend(urls)

        clean_urls.append(urls)
        for url in urls:
            if group[2] == "IO":
                if url not in io_urls_map:
                    io_urls_map[url] = [group[0]]
                else:
                    io_urls_map[url].append(group[0])
            else:
                if url not in control_urls_map:
                    control_urls_map[url] = [group[0]]
                else:
                    control_urls_map[url].append(group[0])


intersec = set(io_urls).intersection(set(control_urls))
urls_to_take = list(intersec)

from tqdm import tqdm

control_users = df[df['group'] == "Control"]['userid'].unique()

df = df[df['urls'].apply(lambda x: len(x) > 0)]
df['urls'] = clean_urls

df_exploded = df.explode('urls')

exposed_adopted_users = []
i = 0

for url in tqdm(urls_to_take):

    # --- FAST URL FILTER
    mask_url = df_exploded['urls'] == url
    df_h = df_exploded[mask_url]

    # Precompute shared masks
    mask_control = df_h['group'] == "Control"
    mask_not_control_io = df_h['interaction_type'].isin(["Control->Control", "Unknown"])  # != "Control->IO"
    mask_io_control = df_h['interaction_type'] == "IO->Control"
    mask_control_control = df_h['interaction_type'] == "Control->Control"
    mask_not_retweet_only = df_h['interaction_form'] != "retweet_only"

    # --- Control users exposed for this url
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
            exposed_adopted_users.append((c_user, url, exposure_io_count, exposure_cc_count,
                                          adopted_count_before_first_io_exposure, adopted_count_after_first_io_exposure,
                                          adopted_count_before_last_io_exposure, adopted_count_after_last_io_exposure,
                                          adopted_count_before_first_cc_exposure, adopted_count_after_first_cc_exposure,
                                          adopted_count_before_last_cc_exposure, adopted_count_after_last_cc_exposure))

    # print(len(exposed_adopted_users))
    if len(exposed_adopted_users) > 100000:
        exposed_adopted_users_df = pd.DataFrame(exposed_adopted_users,
                                                columns=["userid", "url", "NumExposureFromIO",
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
            exposed_adopted_users_df.to_csv("data/{}/urls_exposed_adopted_users_full_{}.csv.gz".format(dataset, i),
                                            index=False, compression="gzip")
        else:
            exposed_adopted_users_df.to_csv("data/{}/urls_exposed_adopted_users_{}.csv.gz".format(dataset, i),
                                            index=False, compression="gzip")
        print("Saved", i)
        i += 1

        del exposed_adopted_users
        exposed_adopted_users = []

exposed_adopted_users_df = pd.DataFrame(exposed_adopted_users,
                                        columns=["userid", "url", "NumExposureFromIO",
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

exposed_adopted_users_df.to_csv("data/{}/urls_exposed_adopted_users_full_{}.csv.gz".format(dataset, i),
                                    index=False, compression="gzip")

print("Saved", i)
