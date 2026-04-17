import pandas as pd
import numpy as np
import datetime
import sys
from utils import load_data

dataset = sys.argv[1]

io_df, control_df = load_data(dataset)

io_users = set(io_df['userid'].unique())
control_users = set(control_df['userid'].unique())
print(dataset)
# Load IO and Control datasets

print(f"Unique IO users: {len(io_users)}")
print(f"Unique Control users: {len(control_users)}")

f = open("data/{}/info_full.txt".format(dataset), "a")
f.write("IO users: {}\n".format(len(io_users)))
f.write("Control users: {}\n".format(len(control_users)))
f.close()

print(f"Overlapping users: {len(io_users.intersection(control_users))}")
# %%
print("Anomaly rate", len(io_users) / (len(io_users) + len(control_users)))
# %%
# Combine datasets and identify replies
# Add labels to identify which group each tweet belongs to
io_df['group'] = 'IO'
control_df['group'] = 'Control'

# Combine datasets
combined_df = pd.concat([io_df, control_df], ignore_index=True)

print(f"Total tweets in combined dataset: {len(combined_df)}")

retweets = combined_df[((combined_df['is_retweet'] == True) & (combined_df['quoted_tweet_tweetid'].isna()))]
retweets_tweets_id = retweets['tweetid'].unique()
retweets = retweets[['userid', 'retweet_userid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]

replies = combined_df[~combined_df['in_reply_to_userid'].isna()]
replies = replies[replies['in_reply_to_userid'] != 'None']
replies_tweets_id = replies['tweetid'].unique()
replies = replies[['userid', 'in_reply_to_userid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]

quoted_tweets = combined_df[(~combined_df['quoted_tweet_tweetid'].isna())]
quoted_tweets_id = quoted_tweets['tweetid'].unique()
quoted_tweets = quoted_tweets[['userid', 'quoted_tweet_tweetid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]

user_mentions = combined_df[~combined_df['user_mentions'].isna()]
user_mentions['user_mentions'] = user_mentions['user_mentions'].apply(lambda x: eval(x) if type(x) == str else x)
user_mentions = user_mentions[user_mentions['user_mentions'].map(len) > 1]
user_mentions = user_mentions[user_mentions['user_mentions'] != '[]']
user_mentions_tweets_id = user_mentions['tweetid'].unique()
user_mentions = user_mentions[['userid', 'user_mentions', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]

retweets['interaction_with_userid'] = retweets['retweet_userid']
del retweets['retweet_userid']

replies['interaction_with_userid'] = replies['in_reply_to_userid']
del replies['in_reply_to_userid']

other = combined_df[['tweetid', 'userid']]
other['interaction_with_userid'] = other['userid']
other['tweetid_join'] = other['tweetid']
del other['userid']
del other['tweetid']

quoted_tweets['tweetid_join'] = quoted_tweets['quoted_tweet_tweetid']
del quoted_tweets['quoted_tweet_tweetid']

quoted_tweets_join = quoted_tweets.set_index('tweetid_join').join(other.set_index('tweetid_join'), lsuffix='_caller', rsuffix='_other')
quoted_interactions = quoted_tweets_join[~quoted_tweets_join['interaction_with_userid'].isna()][['userid', 'interaction_with_userid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]
quoted_interactions = quoted_interactions.reset_index()[['userid', 'interaction_with_userid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]

tweets_interactions = list(retweets_tweets_id) + list(replies_tweets_id) + list(quoted_tweets_id) + list(user_mentions_tweets_id)
only_tweets = combined_df[~combined_df['tweetid'].isin(tweets_interactions)]#[['userid', 'group', 'tweet_time']]
only_tweets['interaction_with_userid'] = np.repeat(np.nan, len(only_tweets))
only_tweets = only_tweets[['userid', 'interaction_with_userid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags']]

import ast

interactions = []
for arr in user_mentions.to_numpy():
    source = arr[0]
    if type(arr[1]) == str:
        arr[1] = ast.literal_eval(arr[1])
    for i, destination in enumerate(arr[1]):
        if i >= 1:
            interactions.append([source, destination, arr[2], arr[3], arr[4], arr[5], arr[6]])

interactions_and_tweets_df = pd.DataFrame(interactions, columns=['userid', 'interaction_with_userid', 'group', 'tweet_time', 'tweetid', 'urls', 'hashtags'])

interactions_and_tweets_df['interaction_form'] = np.repeat('mention', len(interactions_and_tweets_df))
retweets['interaction_form'] = np.repeat('retweet_only', len(retweets))
replies['interaction_form'] = np.repeat('reply', len(replies))
quoted_interactions['interaction_form'] = np.repeat('retweet_quote', len(quoted_interactions))
only_tweets['interaction_form'] = np.repeat('tweet', len(only_tweets))

interactions_and_tweets_df = pd.concat((interactions_and_tweets_df, retweets, replies, quoted_interactions, only_tweets))

interactions_and_tweets_df['time'] = interactions_and_tweets_df['tweet_time']
del interactions_and_tweets_df['tweet_time']

# %%
# Create a mapping from userid to group
# We need to identify which group the replied-to user belongs to

user_lookup = combined_df[['userid', 'group']].drop_duplicates()
user_lookup = user_lookup.groupby('userid').agg({
    'group': lambda x: x.value_counts().index[0]  # Most common group
}).reset_index()

# Create a dictionary for quick lookup
userid_to_group = dict(zip(user_lookup['userid'], user_lookup['group']))

print(f"Created lookup for {len(userid_to_group)} user IDs")


# %%
# Classify each reply interaction
def classify_interaction(row):
    """Classify reply as IO->IO, IO->Control, Control->IO, or Control->Control"""
    replier_group = row['group']
    replied_to_userid = row['interaction_with_userid']

    if pd.isna(replied_to_userid):
        return 'Unknown'

    # Look up the replied-to user's group
    replied_to_group = userid_to_group.get(replied_to_userid, 'Unknown')

    if replied_to_group == 'Unknown':
        return 'Unknown'

    # Return the interaction type (who -> whom)
    return f"{replier_group}->{replied_to_group}"

# print(interactions_and_tweets_df['interaction_with_userid'])
interactions_and_tweets_df['userid'] = interactions_and_tweets_df['userid'].astype("string")
interactions_and_tweets_df['interaction_with_userid'] = interactions_and_tweets_df['interaction_with_userid'].astype("string")
interactions_and_tweets_df['tweetid'] = interactions_and_tweets_df['tweetid'].astype("string")

interactions_and_tweets_df['interaction_type'] = interactions_and_tweets_df.apply(classify_interaction, axis=1)

interactions_and_tweets_df = interactions_and_tweets_df.drop_duplicates(subset=['userid', 'interaction_with_userid', 'group', 'time', 'tweetid'])

# Count interactions
interaction_counts = interactions_and_tweets_df['interaction_type'].value_counts()
print("\nReply Interaction Counts:")
print(interaction_counts)

# %%
# Calculate proportions for each interaction type
total_interactions = len(interactions_and_tweets_df[interactions_and_tweets_df['interaction_type'] != 'Unknown'])

# Get counts for each interaction
control_to_control = interaction_counts.get('Control->Control', 0)
control_to_io = interaction_counts.get('Control->IO', 0)
io_to_control = interaction_counts.get('IO->Control', 0)
io_to_io = interaction_counts.get('IO->IO', 0)

print(f"\nTotal classified interactions: {total_interactions}")
print(f"Control -> Control: {control_to_control} ({control_to_control / total_interactions:.2%})")
print(f"Control -> IO: {control_to_io} ({control_to_io / total_interactions:.2%})")
print(f"IO -> Control: {io_to_control} ({io_to_control / total_interactions:.2%})")
print(f"IO -> IO: {io_to_io} ({io_to_io / total_interactions:.2%})")

f = open("data/{}/info_full.txt".format(dataset), "a")
f.write("Total classified replies\n")
f.write("Control -> Control: {}\n".format(control_to_control))
f.write("Control -> IO: {}\n".format(control_to_io))
f.write("IO -> IO: {}\n".format(io_to_io))
f.write("IO -> Control: {}\n".format(io_to_control))
f.close()

# %%
# Calculate normalized proportions (proportion of replies from each group)
# For Control users: what proportion goes to Control vs IO
control_interactions = control_to_control + control_to_io
if control_interactions > 0:
    control_to_control_prop = control_to_control / control_interactions
    control_to_io_prop = control_to_io / control_interactions
else:
    control_to_control_prop = 0
    control_to_io_prop = 0

# For IO users: what proportion goes to Control vs IO
io_interactions = io_to_control + io_to_io
if io_interactions > 0:
    io_to_control_prop = io_to_control / io_interactions
    io_to_io_prop = io_to_io / io_interactions
else:
    io_to_control_prop = 0
    io_to_io_prop = 0

print("\nNormalized Proportions (within each group):")
print(f"Organic -> Organic: {control_to_control_prop:.3f}")
print(f"Organic -> IO: {control_to_io_prop:.3f}")
print(f"IO -> Organic: {io_to_control_prop:.3f}")
print(f"IO -> IO: {io_to_io_prop:.3f}")

interactions_df = interactions_and_tweets_df[~interactions_and_tweets_df['interaction_with_userid'].isna()]

interactions_df_compacted_motif = interactions_df[['userid', 'interaction_with_userid', 'time', 'interaction_type', 'interaction_form']]
interactions_df_compacted_motif = interactions_df_compacted_motif[~((interactions_df_compacted_motif['interaction_form'] != 'tweet') & (interactions_df_compacted_motif['interaction_type'] == "Unknown"))]

interactions_df_compacted_motif_io_io = interactions_df_compacted_motif[interactions_df_compacted_motif['interaction_type'] == "IO->IO"] #interactions_df_compacted_motif_io_all[interactions_df_compacted_motif_io_all['interaction_with_userid'].isin(io_users)]
interactions_df_compacted_motif_control_control = interactions_df_compacted_motif[interactions_df_compacted_motif['interaction_type'] == "Control->Control"]#interactions_df_compacted_motif_control_all[interactions_df_compacted_motif_control_all['interaction_with_userid'].isin(control_users)]
interactions_df_compacted_motif_io_control = interactions_df_compacted_motif[interactions_df_compacted_motif['interaction_type'] == "IO->Control"]#interactions_df_compacted_motif_io_all[interactions_df_compacted_motif_io_all['interaction_with_userid'].isin(control_users)]
interactions_df_compacted_motif_io_and_control = interactions_df_compacted_motif[interactions_df_compacted_motif['interaction_type'].isin(["IO->Control", "IO->IO"])]#interactions_df_compacted_motif_io_all[((interactions_df_compacted_motif_io_all['interaction_with_userid'].isin(control_users)) | (interactions_df_compacted_motif_io_all['interaction_with_userid'].isin(io_users)))]

interactions_df_compacted_motif_io_io = interactions_df_compacted_motif_io_io.sort_values('time')
interactions_df_compacted_motif_control_control = interactions_df_compacted_motif_control_control.sort_values('time')
interactions_df_compacted_motif_io_control = interactions_df_compacted_motif_io_control.sort_values('time')
interactions_df_compacted_motif_io_and_control = interactions_df_compacted_motif_io_and_control.sort_values('time')

interactions_df_compacted_motif_io_io.to_csv('data/{}/io_io_interactions_graph_full.csv'.format(dataset), index=False)
interactions_df_compacted_motif_control_control.to_csv(
    'data/{}/control_control_interactions_graph_full.csv'.format(dataset), index=False)
interactions_df_compacted_motif_io_control.to_csv('data/{}/io_control_interactions_graph_full.csv'.format(dataset),
                                                  index=False)
interactions_df_compacted_motif_io_and_control.to_csv(
    'data/{}/io_and_control_interactions_graph_full.csv'.format(dataset), index=False)

interactions_and_tweets_df.to_pickle('data/{}/interactions_and_tweets_graph_full.pkl'.format(dataset))
interactions_df.to_csv('data/{}/interactions_graph_full.csv'.format(dataset), index=False)
