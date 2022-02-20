import pandas as pd


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


def train_data():
    dfs = []
    for year in [str(a) for a in range(2010, 2022)]:
        df = pd.read_csv(f'data/raw/{year}.csv', lineterminator='\n').fillna('')
        df_us = df[df['country'] == 'United States']
        df_sample = df_us[df_us['county'] == ''].sample(5000)
        dfs.append(df_sample)

    df_train = pd.concat(dfs, ignore_index=True)

    df_train['baseline_label'] = df_train.text.str.lower().replace('#', '').str.contains('|'.join(all_keywords))
    df_train['text'] = df_train.text.str.replace('\n', ' ').str.replace('\r', ' ')
    df_train.to_csv('data/NLP_train.csv')


if __name__ == "__main__":
    train_data()