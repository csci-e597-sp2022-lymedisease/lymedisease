import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# df = pd.read_csv('data/carmen_tweets_350k.csv', lineterminator='\n').fillna('')
# df_count = df.groupby('county').size().reset_index(name='counts')
# counties_low = list(df_count[df_count.counts < 300].sample(10).county)
# counties_high = list(df_count[df_count.counts >= 300].sample(10).county)
#
# print(counties_low)
# print(counties_high)

counties = ['Chester County', 'Westmoreland County', 'Ocean County', 'Fairfield County', 'Chittenden County',
            'Bristol County', 'Hampshire County', 'Erie County', 'Burlington County', 'Schenectady County']


# df_tweets = pd.read_csv('data/carmen_tweets_350k.csv', lineterminator='\n').fillna('')
# df_tweets['year'] = df_tweets.created_at.map(lambda x: int(str(x)[0:4]))
#
# cols = ['Ctyname'] + ['Cases' + str(a) for a in range(2010, 2020)]
# df_lyme = pd.read_csv('data/LD-Case-Counts-by-County-00-19.csv', usecols=cols).set_index('Ctyname')


def get_county_tweets(county, df_tweets):
    df_county = df_tweets[df_tweets['county'] == county]
    counts = dict(df_county.groupby('year').size())
    df_year_counts = pd.DataFrame(0, index=range(2010, 2020), columns=['tweet_count'])
    for k, v in counts.items():
        if k in range(2010, 2020):
            df_year_counts.loc[k, 'tweet_count'] = v
    return df_year_counts


def get_county_cases(county, df_lyme):
    county_cases = df_lyme.loc[county,]
    if isinstance(county_cases, pd.DataFrame):
        cases = list(county_cases.iloc[0])
    else:
        cases = list(county_cases)
    # print(cases)
    return cases


def plot_tweets_cases(county, df_year_counts, cases):
    df_year_counts['cases'] = cases
    plt.scatter('tweet_count', 'cases', data=df_year_counts)
    plt.title(county)
    plt.xlabel('tweet counts')
    plt.ylabel('cases')
    # plt.savefig(f'{county}.png')
    plt.show()


def fit_model(df_year_counts, cases):
    df_year_counts['cases'] = cases
    print(df_year_counts)
    model = LinearRegression()
    X = df_year_counts.tweet_count.values.reshape(-1, 1)
    y = df_year_counts.cases.values.reshape(-1, 1)
    model.fit(X, y)
    return model.score(X, y)



if __name__ == "__main__":
    df_tweets = pd.read_csv('data/carmen_tweets_keywords_pos.csv', lineterminator='\n').fillna('')
    df_tweets['year'] = df_tweets.created_at.map(lambda x: int(str(x)[0:4]))

    cols = ['Ctyname'] + ['Cases' + str(a) for a in range(2010, 2020)]
    df_lyme = pd.read_csv('data/LD-Case-Counts-by-County-00-19.csv', usecols=cols).set_index('Ctyname')

    # for county in counties:
    #     df_year_counts = get_county_tweets(county, df_tweets)
    #     cases = get_county_cases(county, df_lyme)
    #     plot_tweets_cases(county, df_year_counts, cases)
    r2s = []

    for county in counties:
        df_year_counts = get_county_tweets(county, df_tweets)
        cases = get_county_cases(county, df_lyme)
        r2 = fit_model(df_year_counts, cases)
        r2s.append(r2)

    print(r2s)
