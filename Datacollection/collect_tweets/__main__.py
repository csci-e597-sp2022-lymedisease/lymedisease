import pandas as pd
import requests
import os
import json
import time
from dotenv import load_dotenv
from pandas.io.json import json_normalize


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    # print(bearer_token)
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    # print(r.headers["Authorization"])
    return r

def connect_to_endpoint(url, params):
    try:
        response = requests.get(url, auth=bearer_oauth, params=params).json()
        # b_errorstatus = False
        global df_tweets, df_places
        df_tweets = df_tweets.append(json_normalize(response, record_path=['data']))
        df_places = df_places.append(json_normalize(response, record_path=[['includes', 'places']]))
        if ('next_token' in response['meta']):
            time.sleep(1)
            params['next_token'] = response['meta']['next_token']
            connect_to_endpoint(url, params)


    except Exception as e:
        print('Error has occured! Error code : ' + str(response['status']) + ', Error descr is ' + response['title'])
        # b_errorstatus = True
        return True
    return False

def geo_tagged_tweets():

    search_url = "https://api.twitter.com/2/tweets/search/all"
    query_params = {
        'query': '(lyme OR #lymedisease OR #lyme) lang:en place_country:us (-place:00627b884512b296 -place:014459087dde32e5)'
        , 'start_time': '2021-12-15T00:00:00.000Z'
        , 'end_time': '2021-12-31T00:00:00.000Z'
        , 'expansions': 'geo.place_id'
        , 'tweet.fields': 'author_id,created_at,in_reply_to_user_id'
        , 'max_results': 50
    }


    b_errorstatus = connect_to_endpoint(search_url, query_params)

    if b_errorstatus is False:
        df_tweets.to_csv('lyme_tweets_100121_123121.csv')
        df_places.to_csv('lyme_places_100121_123121.csv')
        print('tweets csv files creation is complete.')


if __name__ == "__main__":
    load_dotenv()
    bearer_token = (os.environ.get('BEARER_TOKEN'))
    df_tweets = df_places = pd.DataFrame()

    geo_tagged_tweets()

