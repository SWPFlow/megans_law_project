import matplotlib.pyplot as plt
from data_pipeline import ListSplitter, RaceDummies, CrimeAndSentence, FeatureEngineer, ColumnFilter
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

def plot_comparison(X, y, column):
    plt.hist(X[y == 1][column], bins = np.linspace(0, X[column].max(), 20))
    plt.hist(X[y == 0][column], bins = np.linspace(0, X[column].max(), 20), alpha=.5)
    print column, y[X[column] >= 1].sum() * 1.0 / y[X[column] == 1].count()
    print 'Not', column, y[X[column] == 0].sum() * 1.0 / y[X[column] == 0].count()
    plt.show()

def do_ttest(X, y, column):
    print column
    a = y[X[column] >= 1]
    b = y[X[column] == 0]
    print ttest_ind(a, b, equal_var=False)


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
    for race in X.columns[1:19]:
        plot_comparison(X, y, race)
    for crime in X.columns[21:-9]:
        # plot_comparison(X, y, crime)
        # do_ttest(X, y, crime)
        pass
