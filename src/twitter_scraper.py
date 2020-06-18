import sys
import tweepy #https://github.com/tweepy/tweepy
import csv
import time
import os
import pandas as pd
import numpy as np

# Twitter and Dropbox API credentials
import api_cred as ac

from processing import logger

sys.path.insert(1, 'drive/My Drive/tweets_data_collection')

# Gets all the authentication to start scraping from Twitter
def authenticate_twitter():
    """
    Gets all the authentication to start scraping from Twitter
    This is dependent on the file api_cred.py. Right now it's
    someone's API credential that was previously on the project.
    He said he doesn't mind, but lol the next user should update
    it by signing up here: https://developer.twitter.com/app/new

    returns: Tweepy API
    """
    auth = tweepy.OAuthHandler(ac.consumer_key, ac.consumer_secret)
    auth.set_access_token(ac.access_key, ac.access_secret)
    # Waiting important because if you're doing a fresh scrape, there
    # will be too many requests sent through Tweepy and it'll crash
    # Keep this to be True or suffer the consequences : - (
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    return api

# Possible tags found in the JSON data per tweet
possible_info = {'contributors', 'coordinates', 'created_at', 'entities',
                 'favorite_count', 'favorited', 'full_text', 'geo', 'id',
                 'id_str', 'in_reply_to_screen_name', 'in_reply_to_status_id',
                 'in_reply_to_status_id_str', 'in_reply_to_user_id',
                 'in_reply_to_user_id_str', 'is_quote_status', 'lang',
                 'place', 'possibly_sensitive', 'retweet_count',
                 'retweeted', 'retweeted_status', 'source', 'text',
                 'truncated', 'user'}

def get_new_tweets(tweet_name, since_id=1, api=authenticate_twitter(),
                   extended_mode=True,
                   info_tags={'id', 'created_at', 'full_text', 'retweet_count',
                              'favorite_count'}):
    """
    Grabs all the tweets since the last tweet loaded. Since_id will be the id
    of the last tweet. When grabbing tweets, we can only grab a max of 200 at a
    time. You can see this in the code below.

    To go around that, we use "max-id", which will be the maximum id of the
    latest tweet loaded at a time.

    The newer the tweet, the larger id value is. So we maintain the "max_id"
    as the last tweet posted by the Twitter handle (tweet_name), and get the
    tweets in reverse from the latest one up until our last taken tweet.

    Example:
            If since_id == 4 and the latest tweets were from 300, we would get
            the tweets from 300 - 101, then 100 - 5. And the functionality of
            grabbing the tweets excludes the since_id.

    params: tweet_name - Twitter handle
            since_id - ID of the latest tweet seen so far
            api - Authentification of Twitter Account
            extended_mode - Setting for grabbing complete Tweet
            info_tags - json tags desired to keep track of
    """
    # Check if we can access account
    try:
        if api.get_user(tweet_name).protected:
            print('Tried to get from @%s, but account protected.'%(tweet_name))
            return None
    except tweepy.TweepError as e:
        print('Tried to get from @%s, but got a "%s" error' % (tweet_name, e))
        return None

    curr_info = possible_info.intersection(info_tags)
    if extended_mode:
        curr_info.add('full_text')
        tweet_mode = 'extended'
    else:
        curr_info.add('text')
        tweet_mode = 'compatibility'
    curr_info.add('id') # Necessary regardless
    curr_info = sorted(curr_info)
    tweets = np.array([])

    new_tweets = api.user_timeline(screen_name = tweet_name,
                                   since_id = since_id,
                                   count=200,
                                   tweet_mode = tweet_mode
                                   )
    while new_tweets:
        new_tweets = get_tweet_info(new_tweets, curr_info)
        tweets = np.append(tweets, new_tweets)
        max_id = new_tweets[-1]['id'] - 1
        new_tweets = api.user_timeline(screen_name = tweet_name,
                                       since_id = since_id,
                                       max_id = max_id,
                                       count = 200,
                                       tweet_mode = tweet_mode
                                       )
    logger()
    print("Downloading %d tweets from %s" % (len(tweets), tweet_name))
    # Reverses back to chronological order
    return list(tweets[::-1])

def get_necessary_info(tweet_status, info_list):
    """
    Helper function fpr get_tweet_info

    params: tweet_status - JSON data of a tweet as a dictionary
            info_list - Specific features in the JSON data
    """
    res_info = {}
    for key in info_list:
        if key != 'full_text' or 'retweeted_status' not in tweet_status:
            res_info[key] = tweet_status[key]
        else:
            res_info[key] = \
                    'RT @{}: {}'.format(
                        tweet_status['retweeted_status']['user']['screen_name'],
                        tweet_status['retweeted_status'][key])
    return res_info

def get_tweet_info(tweet_statuses, info_list):
    """
    Grabs all the necessary json info and places them into a nice little
    dictionary.

    params: tweet_statuses - JSON data of multiple tweets
            info_list - Specific features in the JSON data
    """
    tweet_statuses = pd.Series(tweet_statuses).apply(lambda x: x._json)
    curr_tweets_series = tweet_statuses.apply(get_necessary_info,
            args=(info_list,))
    return curr_tweets_series.to_numpy()

def df_to_file(row, out_fp):
    """
    Writes each row directly to file

    params: row - Twitter Handle and list of tweets
            out_fp - file output path
    """
    # Edge case where first row is repeated in apply function
    global first_row
    if first_row:
        first_row = False
        return

    curr_handle = row['handle']
    # Edge case when no tweets found
    if not row['tweet_list']:
      return {'handle': curr_handle,
              'since_id': 1}

    # Converts to dataframe with each tag and Twitter handle as a column
    curr_handle_tweets = pd.DataFrame(row['tweet_list'])
    curr_cols = curr_handle_tweets.columns.tolist()
    curr_handle_tweets['handle'] = curr_handle

    # Rearranges so that handle is the first column
    curr_handle_tweets = curr_handle_tweets[['handle'] + sorted(curr_cols)]

    # Appends to file
    curr_handle_tweets.to_csv(out_fp, mode='a', header=False, index=False)

    # Returns Twitter handle and most recent pulled id
    return {'handle': curr_handle,
            'since_id': curr_handle_tweets['id'].iloc[-1]}

def get_data(tweet_output_fp, twitter_handles_fp, handles_record_fp,
                 tags={'id', 'created_at', 'full_text', 'retweet_count',
                     'favorite_count'},
                 chunk=100, extended_mode=True):
    """
    Pulls new latest tweets and appends them to the correct csv file
    params: tweet_output_fp - path to the file to save the new tweets
            twitter_handles_fp - path to file that contain the twitter handles,
                                 last tweet pulled id, tweet counts, and
                                 last pull date
            handles_record_fp - path to record the last requested tweet
            tags - JSON tags to be stored into csv
            chunk - how many accounts we scrape at a time
    """

    global first_row

    print("Start...")

    # process the paths so they are passable to load_sheets
    tweets_path = os.path.expanduser(tweet_output_fp)
    twitter_list_path = os.path.expanduser(twitter_handles_fp)

    # load and prepare list of twitter accounts
    list_df = pd.read_csv(twitter_list_path)

    # API Key
    api = authenticate_twitter()

    # Writes headers of output and records files
    with open(tweet_output_fp, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['handle'] + sorted(tags))
    with open(handles_record_fp, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['handle', 'id'])

    already_seen = set()
    for col in list_df.columns:
        if 'Twitter Handle' in col:
            for split in range(0, list_df.shape[0], chunk):
                curr_handles = list_df[col].iloc[split:split + chunk].dropna()
                if len(already_seen.intersection(curr_handles) - {' '}):
                    print('Intersection',
                          already_seen.intersection(curr_handles) - {' '})
                # Gets only the handles not seen yet
                curr_handles = \
                        pd.Series(list(set(curr_handles) - already_seen))
                if not curr_handles.shape[0]:
                    continue
                # Gets tweets
                curr_tweets_series = curr_handles.apply(get_new_tweets,
                                                        args=(1, api,
                                                              extended_mode,
                                                              tags))
                # Convert to format for df_to_file
                curr_tweets_series = curr_tweets_series.rename('tweet_list')
                curr_tweets_series.index = curr_handles
                curr_tweets_df = curr_tweets_series.reset_index()
                curr_tweets_df.rename(columns={'index': 'handle'},
                        inplace=True)
                # first_row to avoid repeating first row
                first_row = True
                max_id_series = curr_tweets_df.apply(df_to_file,
                                                 args=(tweets_path,),
                                                 axis=1)
                max_id_series[0] = df_to_file(curr_tweets_df.iloc[0],
                                          tweets_path)
                # Get the max_id for the new since_id value
                max_id_df = \
                        pd.DataFrame(max_id_series.tolist()).drop_duplicates()
                max_id_df.to_csv(handles_record_fp, mode='a',
                                 header=False, index=False)
                already_seen = already_seen.union(curr_handles)
                print('Done up to %s' % (curr_handles.iloc[-1]))
