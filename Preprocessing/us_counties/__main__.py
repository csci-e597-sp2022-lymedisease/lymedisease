import pandas as pd
import os


def get_county_data(year, filepath):
    df = pd.read_csv(f'data/raw/{year}.csv', lineterminator='\n').fillna('')
    df_us = df[df['country'] == 'United States']
    df_county = df_us[df_us['county'] != ''].copy()
    df_county['text'] = df_county['text'].map(lambda x: x.replace('\r', ''))
    # relevant_df = df_county[df_county.text.str.lower().replace('#', '').str.contains('|'.join(keywords))].copy()
    # relevant_df['text'] = relevant_df['text'].map(lambda x: x.replace('\r', ''))
    # relevant_df.to_csv(f'data/us_county/{year}_county.csv', index=False)
    if not os.path.exists(filepath):
        df_county.to_csv(filepath, index=False, header=True)
    else:
        df_county.to_csv(filepath, index=False, header=False, mode='a')



if __name__ == "__main__":
    for i in [str(a) for a in range(2010, 2022)]:
        get_county_data(i, 'data/us_county.csv')
