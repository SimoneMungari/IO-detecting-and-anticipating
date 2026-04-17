import pandas as pd
from nltk.corpus import stopwords, nonbreaking_prefixes
import nltk
import re
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
        return utcdttime
    except:
        return None


def process_data(tweet_df):
    tweet_df['quoted_tweet_tweetid'] = tweet_df['quoted_tweet_tweetid'].astype('Int64')
    tweet_df['retweet_tweetid'] = tweet_df['retweet_tweetid'].astype('Int64')

    # Tweet type classification
    tweet_type = []
    for i in range(tweet_df.shape[0]):
        if pd.notnull(tweet_df['quoted_tweet_tweetid'].iloc[i]):
            if pd.notnull(tweet_df['retweet_tweetid'].iloc[i]):
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    continue
                else:
                    tweet_type.append('retweet')
            else:
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    tweet_type.append('reply')
                else:
                    tweet_type.append('quoted')
        else:
            if pd.notnull(tweet_df['retweet_tweetid'].iloc[i]):
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    continue
                else:
                    tweet_type.append('retweet')
            else:
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    tweet_type.append('reply')
                else:
                    tweet_type.append('original')
    tweet_df['tweet_type'] = tweet_type
    tweet_df = tweet_df[tweet_df.tweet_type != 'retweet']

    return tweet_df


def get_positive_data(pos_df):
    pos_df = process_data(pos_df)
    pos_df = pos_df[['tweetid', 'userid', 'tweet_time', 'tweet_language', 'tweet_text']]
    pos_df['tweet_time'] = pos_df['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    return pos_df


# Downloading Stopwords
nltk.download('stopwords')
nltk.download('nonbreaking_prefixes')

# Load English Stop Words
languages = ['arabic',
 'english',
 'indonesian',
 'russian']
stopword = stopwords.words(languages)


def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['tweet_text'] = df['tweet_text'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['tweet_text'] = df['tweet_text'].str.replace(":", '')
    df['tweet_text'] = df['tweet_text'].str.replace(";", '')
    df['tweet_text'] = df['tweet_text'].str.replace(".", '')
    df['tweet_text'] = df['tweet_text'].str.replace(",", '')
    df['tweet_text'] = df['tweet_text'].str.replace("!", '')
    df['tweet_text'] = df['tweet_text'].str.replace("&", '')
    df['tweet_text'] = df['tweet_text'].str.replace("-", '')
    df['tweet_text'] = df['tweet_text'].str.replace("_", '')
    df['tweet_text'] = df['tweet_text'].str.replace("$", '')
    df['tweet_text'] = df['tweet_text'].str.replace("/", '')
    df['tweet_text'] = df['tweet_text'].str.replace("?", '')
    df['tweet_text'] = df['tweet_text'].str.replace("''", '')
    # Lowercase
    df['tweet_text'] = df['tweet_text'].str.lower()

    return df


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# Message Clean Function
def msg_clean(msg):
    # Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    # Remove Mentions
    msg = re.sub(r'@\w+', ' ', msg)

    # Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove Emoji from text
    msg = remove_emoji(msg)

    # Remove Stop Words
    msg = msg.split()

    msg = " ".join([word for word in msg if word not in stopword])

    return msg

