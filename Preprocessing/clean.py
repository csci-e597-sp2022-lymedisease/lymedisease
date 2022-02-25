import pandas as pd


def check_county_match():
    df_tweets = pd.read_csv('data/carmen_tweets_350k.csv', lineterminator='\n').fillna('')
    tweet_counties = {f'{c},{s}' for c, s in zip(df_tweets.county, df_tweets.state)}

    df_lyme = pd.read_csv('data/LD-Case-Counts-by-County-00-19.csv')
    lyme_counties = {f'{c},{s}' for c, s in zip(df_lyme.Ctyname, df_lyme.Stname)}

    mismatch = []
    for c in tweet_counties:
        if c not in lyme_counties:
            mismatch.append(c)

    return True if not mismatch else mismatch


def match_county(str):
    changes = {'St Clair County': 'St. Clair County', 'St Landry Parish': 'St. Landry Parish',
               'Baltimore City': 'Baltimore city', 'St Joseph County': 'St. Joseph County',
               'La Porte County': 'LaPorte County', 'Dekalb County': 'DeKalb County',
               'Falls Church City': 'Falls Church city', 'Williamsburg City': 'Williamsburg city',
               'St Louis City': 'St. Louis city', 'Norfolk City': 'Norfolk city',
               'St Charles County': 'St. Charles County', 'St Francois County': 'St. Francois County',
               'Martinsville City': 'Martinsville city', 'Winchester City': 'Winchester city',
               'St Johns County': 'St. Johns County', 'Roanoke City': 'Roanoke city',
               'Danville City': 'Danville city', 'Fond Du Lac County': 'Fond du Lac County',
               'St Lucie County': 'St. Lucie County', 'Manassas City': 'Manassas city',
               'Staunton City': 'Staunton city', 'St Louis County': 'St. Louis County',
               'Franklin City': 'Franklin city', 'St Lawrence County': 'St. Lawrence County',
               'Alexandria City': 'Alexandria city', 'Wade Hampton Census Area': 'Kusilvak Census Area',
               'Hampton City': 'Hampton city', 'Virginia Beach City': 'Virginia Beach city',
               'Newport News City': 'Newport News city', 'St Tammany Parish': 'St. Tammany Parish',
               "St Mary's County": "St. Mary's County", 'Fairfax City': 'Fairfax city',
               'Richmond City': 'Richmond city', 'Suffolk City': 'Suffolk city',
               'Waynesboro City': 'Waynesboro city', 'Anchorage Borough': 'Anchorage Municipality',
               'Radford City': 'Radford city', 'Portsmouth City': 'Portsmouth city',
               'Lynchburg City': 'Lynchburg city', 'Juneau Borough': 'Juneau City and Borough',
               'Chesapeake City': 'Chesapeake city', 'Charlottesville City': 'Chesapeake city',
               'Dona Ana County': 'Doña Ana County', 'Harrisonburg City': 'Harrisonburg city',
               'Petersburg City': 'Petersburg city', 'Lexington City': 'Lexington city',
               'Salem City': 'Salem city', 'Fredericksburg City': 'Fredericksburg city',
               'Dupage County': 'DuPage County', 'St Francis County': 'St. Francis County'}

    return changes[str] if str in changes.keys() else str


def match_carmen_to_cdc():
    df_tweets = pd.read_csv('data/carmen_tweets_350k.csv', lineterminator='\n').fillna('')
    df_tweets = df_tweets[-df_tweets['state'].isin(['Northern Mariana Islands', 'Puerto Rico'])]
    df_tweets.loc[(df_tweets.state == 'Illinois') & (df_tweets.county == 'La Salle County'), "county"] = 'LaSalle County'
    df_tweets['county'] = df_tweets.county.map(lambda x: match_county(x))
    df_tweets.drop(columns=['Unnamed: 0'], inplace=True)
    df_tweets.to_csv('data/clean/carmen_tweets_350k.csv', header=True, index=False)


keywords = ['lyme disease', 'lymedisease', 'neuro', 'long-haul', 'long haul', 'have lyme', 'had lyme', 'having lyme',
            'has lyme', 'specialist', 'physician', 'doctor', 'neurologist', 'dermatologist', 'rash', 'flu', 'symptom',
            'fever', 'ache', 'pain', 'hiking', 'hike', 'forest', 'bulls eye', 'bulls-eye', 'bullseye', 'bull\'s eye',
            'bull\'s-eye', 'tick', 'death', 'die', 'red color', 'swollen', 'health', 'medical care', 'med check',
            'medical checkup', 'late stage', 'early stage', 'antibiotic', 'inflammation', 'heart', 'illness', 'deer',
            'get lyme', 'gets lyme', 'got lyme', 'getting lyme', 'diagnose', 'diagnosis', 'patient', 'hospital',
            'clinic', 'cure', 'treat', 'heal', 'disease', 'meds', 'medication', 'medicine', 'therapy', 'infection',
            'lyme\'s', 'tested positive', 'tested negative', 'lyme test', 'lyme\'s test']

keywords_medical = ['tick', 'ticks', 'bite', 'borreliosis', 'zoonotic', 'infection', 'forest', 'tickborne', 'erythema',
                    'migrans', 'carditis', 'neuroborreliosis', 'borrelia', 'bacterium', 'ixodes', 'blackleg',
                    'blacklegged', 'burgdorferi', 'borrelial', 'lymphocytoma', 'arthritis', 'deer', 'deertick', 'fever',
                    'headache', 'headaches', 'paralysis', 'hearing', 'rash', 'fatigue', 'swollen', 'lymph', 'chill',
                    'chills', 'flu', 'sweat', 'inflammatory', 'neck', 'knee', 'knees', 'stiffness', 'heart',
                    'palpitations', 'numbness', 'tingling', 'nausea', 'vomiting', 'neurologic', 'vertigo', 'dizziness',
                    'sleepless', 'fogginess', 'nerve', 'irritability', 'joint', 'depression', 'memory', 'malaise']

all_keywords = keywords + keywords_medical


def add_cols():
    df_tweets = pd.read_csv('data/clean/carmen_tweets_350k.csv', lineterminator='\n').fillna('')
    df_tweets['cty_st'] = [f'{c},{s}' for c, s in zip(df_tweets.county, df_tweets.state)]
    df_tweets['year'] = df_tweets.created_at.map(lambda x: int(str(x)[0:4]))
    df_tweets['fixed_text'] = df_tweets.text.str.lower().replace('#', '')
    df_tweets['keywords_label'] = df_tweets.fixed_text.str.contains('|'.join(all_keywords)).astype(int)
    df_tweets.drop(columns=['fixed_text'], inplace=True)
    df_tweets.to_csv('data/clean/carmen_tweets_350k.csv', header=True, index=False)


def get_us_no_county():
    dfs = []
    for year in [str(a) for a in range(2010, 2022)]:
        df = pd.read_csv(f'data/raw/{year}.csv', lineterminator='\n').fillna('')
        df = df[(df['country'] == 'United States') & (df['county'] == '')]
        df['year'] = df.created_at.map(lambda x: int(str(x)[0:4]))
        df['fixed_text'] = df.text.str.lower().replace('#', '')
        df['keywords_label'] = df.fixed_text.str.contains('|'.join(all_keywords)).astype(int)
        df.drop(columns=['fixed_text'], inplace=True)
        df['keywords_label'] = df.text.str.contains('|'.join(all_keywords)).astype(int)
        df['text'] = df.text.str.replace('\n', ' ').str.replace('\r', ' ')
        dfs.append(df)
    df_all = pd.concat(dfs)
    print(len(df_all))
    df_all.to_csv('data/clean/no_carmen_us_360k.csv', header=True, index=False)



get_us_no_county()