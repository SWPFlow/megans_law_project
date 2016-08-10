import matplotlib.pyplot as plt
from data_pipeline import ListSplitter, RaceDummies, CrimeAndSentence, FeatureEngineer, ColumnFilter
from sklearn.pipeline import Pipeline
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/Megans_List.csv')
    df['Violation'] = df['redtext1'].str.contains('VIOLATION') * 1
    y = df['Violation']
    y.fillna(0, inplace=True)
    p = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter())
    ])

    X = p.fit_transform(df.copy(), y)
