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
    response = requests.get(url, auth=bearer_oauth, params=params).json()
    if 'next_token' in response['meta']:
        next_token = query_params['next_token'] = response['meta']['next_token']
    else:
        next_token = None

    return response


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
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAPIEYwEAAAAAfxb4aUzNobtohOlq01OjM6t9O20%3DS0i50a1d7M2r7uJOWu20IlX9YODRoQCUNQ4xx97rGo0wtvKsuF'

    json_response = connect_to_endpoint(search_url, query_params)
    tweet_count = tweet_count + json_response['meta']['total_tweet_count']

    while ('next_token' in json_response['meta']):
        time.sleep(1)
        json_response = connect_to_endpoint(search_url, query_params)
        tweet_count = tweet_count + json_response['meta']['total_tweet_count']

    print('Total tweets for the given query is ', tweet_count)