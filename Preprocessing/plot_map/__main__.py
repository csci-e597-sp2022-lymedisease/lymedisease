import pandas as pd
from urllib.request import urlopen
import json
import plotly.express as px
from plotly.subplots import make_subplots


def plot_tweets(year):
    df = pd.read_csv(f'data/raw/{year}.csv', lineterminator='\n').fillna('')
    df_us = df[df['country'] == 'United States']
    df_county = df_us[df_us['county'] != '']
    county_count = df_county.groupby('county').size().reset_index(name='count')
    county_count['county'] = county_count['county'].str.replace(' County', '')

    fips_df = pd.read_csv('data/other/fips.csv')
    fips_df['fips_str'] = [str(i) if i >= 10000 else '0' + str(i) for i in fips_df.FIPS]

    plot_df = pd.merge(county_count, fips_df, how='left', left_on='county', right_on='Name')

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth_mapbox(plot_df, geojson=counties, locations='fips_str', color='count', width=600, height=400,
                               color_continuous_scale="Viridis", range_color=(0, 12), mapbox_style="carto-positron",
                               zoom=2.5, center={"lat": 37.0902, "lon": -95.7129}, opacity=0.5,
                               labels={'count': 'tweet count'})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    return fig


def plot_lyme(year):
    df = pd.read_csv('data/LD-Case-Counts-by-County-00-19.csv')
    df_year = df[['Ctyname', f'Cases{year}']].copy()
    df_year['Ctyname'] = df_year['Ctyname'].str.replace(' County', '')
    df_year = df_year[df_year[f'Cases{year}'] != 0]

    fips_df = pd.read_csv('data/other/fips.csv')
    fips_df['fips_str'] = [str(i) if i >= 10000 else '0' + str(i) for i in fips_df.FIPS]

    plot_df = pd.merge(df_year, fips_df, how='left', left_on='Ctyname', right_on='Name')

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth_mapbox(plot_df, geojson=counties, locations='fips_str', color=f'Cases{year}', width=600, height=400,
                               color_continuous_scale="Viridis", range_color=(0, 12), mapbox_style="carto-positron",
                               zoom=2.5, center={"lat": 37.0902, "lon": -95.7129}, opacity=0.5,
                               labels={'count': 'case count'})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    return fig


if __name__ == "__main__":
    for year in [str(a) for a in range(2010, 2022)]:
        fig_tweets = plot_tweets(year)
        fig_tweets.write_image(f'plots/{year}_tweets.png')
        fig_lyme = plot_lyme(year)
        fig_lyme.write_image(f'plots/{year}_lyme.png')