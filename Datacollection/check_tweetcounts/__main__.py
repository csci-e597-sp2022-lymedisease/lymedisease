import pandas as pd
import requests
import os
import json
import time

from dotenv import load_dotenv


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"

    return r


def connect_to_endpoint(url, params):
    try:
        response = requests.get(url, auth=bearer_oauth, params=params).json()
        global tweet_count
        tweet_count = tweet_count + response['meta']['total_tweet_count']
        if ('next_token' in response['meta']):
            # time.sleep(1)
            params['next_token'] = response['meta']['next_token']
            connect_to_endpoint(url, params)

    except Exception as e:
        print('Error has occured! Error code : ' + str(response['status']) + ', Error descr is ' + response['title'])

    return tweet_count


if __name__ == "__main__":

    load_dotenv()
    bearer_token = (os.environ.get('BEARER_TOKEN'))
    query_params = {'query': '(lyme OR #lyme OR #lymedisease ) lang:en -is:retweet (-place:00627b884512b296 -place:014459087dde32e5)'
        , 'start_time': '2021-10-01T00:00:00.000Z'
        , 'end_time': '2021-12-31T00:00:00.000Z'
                    }
    search_url = "https://api.twitter.com/2/tweets/counts/all"
    next_token = ''
    tweet_count = 0

    tweet_count = connect_to_endpoint(search_url, query_params)

    print('Total tweets for the given query is ', tweet_count)