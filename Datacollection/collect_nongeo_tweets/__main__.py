import os
from dotenv import load_dotenv
import requests
import pandas as pd
import carmen
import time


def create_headers():
    load_dotenv()
    return {"Authorization": f"Bearer {os.getenv('Bearer_Token')}"}


def create_url(keyword, start_time, end_time, max_results=10):
    search_url = "https://api.twitter.com/2/tweets/search/all"
    query_params = {'query': keyword,
                    'start_time': start_time,
                    'end_time': end_time,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,conversation_id,created_at,entities,'
                                    'public_metrics,referenced_tweets,reply_settings,source,context_annotations',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,location',
                    # 'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return search_url, query_params


def connect_to_endpoint(url, headers, params, next_token=None):
    params['next_token'] = next_token
    response = requests.request("GET", url, headers=headers, params=params)
    # print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def resolve_location(tweets, users_df):
    resolver = carmen.get_resolver()
    resolver.load_locations()

    for tweet in tweets:
        user = tweet['author_id']
        tweet['user'] = dict(users_df.loc[user, ])

        location = resolver.resolve_tweet(tweet)

        if location:
            tweet['city'] = location[1].city
            tweet['state'] = location[1].state
            tweet['county'] = location[1].county
            tweet['country'] = location[1].country
        else:
            continue
    return tweets


def append_to_csv(tweets, filepath, header=False):
    for tweet in tweets:
        tweet['user_name'] = tweet['user']['username']
    df = pd.DataFrame(tweets, columns=['id', 'text', 'author_id', 'public_metrics', 'created_at', 'in_reply_to_user_id',
                                       'conversation_id', 'reply_settings', 'source', 'referenced_tweets',
                                       'context_annotations', 'entities', 'user', 'user_name', 'city', 'state',
                                       'county', 'country'])
    if header:
        df.to_csv(filepath, header=True, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)


def collect_data(year, keywords):
    fp = f'{year}.csv'
    query = f'({keywords}) lang:en -has:geo -is:retweet'
    start_time = f'{year}-01-01T00:00:00.000Z'
    end_time = f'{year}-12-31T23:59:59.000Z'
    headers = create_headers()
    url, params = create_url(query, start_time, end_time, max_results=100)

    count = 0
    finish = False
    next_token = None

    while not finish:
        js = connect_to_endpoint(url, headers, params, next_token)
        users_data = pd.DataFrame(js['includes']['users']).set_index('id').fillna('')
        tweets = js['data']
        tweets_w_loc = resolve_location(tweets, users_data)
        if count == 0:
            append_to_csv(tweets_w_loc, fp, header=True)
        else:
            append_to_csv(tweets_w_loc, fp, header=False)

        count += js['meta']['result_count']

        if 'next_token' in js['meta']:
            next_token = js['meta']['next_token']
            print(next_token)
        else:
            finish = True
        time.sleep(4)

    print(f'----- Total Count for year {year} is {count}. -----')



if __name__ == "__main__":
    for year in [str(a) for a in range(2010, 2022)]:
        collect_data(year, 'lyme OR #lymedisease OR #lyme')


