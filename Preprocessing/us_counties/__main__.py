import pandas as pd


def get_county_data(year):
    df = pd.read_csv(f'{year}.csv', lineterminator='\n').fillna('')
    df_us = df[df['country'] == 'United States']
    df_county = df_us[df_us['county'] != '']
    relevant_df = df_county[df_county.text.str.lower().replace('#', '').str.contains('|'.join(keywords))].copy()
    relevant_df['text'] = relevant_df['text'].map(lambda x: x.replace('\r', ''))
    relevant_df.to_csv(f'data/us_county/{year}_county.csv', index=False)


keywords = ['lyme disease', 'lymedisease', 'neuro', 'long-haul', 'long haul', 'have lyme', 'had lyme', 'having lyme',
            'has lyme', 'specialist', 'physician', 'doctor', 'neurologist', 'dermatologist', 'rash', 'flu', 'symptom',
            'fever', 'ache', 'pain', 'hiking', 'hike', 'forest', 'bulls eye', 'bulls-eye', 'bullseye', 'bull\'s eye',
            'bull\'s-eye', 'tick', 'death', 'die', 'red color', 'swollen', 'health', 'medical care', 'med check',
            'medical checkup', 'late stage', 'early stage', 'antibiotic', 'inflammation', 'heart', 'illness', 'deer',
            'get lyme', 'gets lyme', 'got lyme', 'getting lyme', 'diagnose', 'diagnosis', 'patient', 'hospital',
            'clinic', 'cure', 'treat', 'heal', 'disease', 'meds', 'medication', 'medicine', 'therapy', 'infection',
            'lyme\'s', 'tested positive', 'tested negative', 'lyme test', 'lyme\'s test']


if __name__ == "__main__":
    for i in [str(a) for a in range(2010, 2022)]:
        df = pd.read_csv(f'data/us_county/{i}_county.csv', lineterminator='\n')
        # print(len(df))
