from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from statsmodels.discrete.discrete_model import Logit
import pandas as pd

def tag_counter(lst, tag_values):
    tot = 0
    for val in tag_values:
        for descrip in lst:
            if val in descrip:
                tot += 1
    return tot

class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])

class FeatureEngineer(CustomMixin):
    code_sentencing = {}
    with open('data/code_sentencing.txt') as f:
        for line in f:
            if 'skip' in line:
                continue
            code = line.split(':')[0]
            sentences = set(float(sentence.strip()) for sentence in line.split(':')[1].split(','))
            code_sentencing[code] = sentences

    descrip_tags = {}
    with open('data/Words_for_Dummies.txt') as f:
        for line in f:
            if 'skip' in line:
                continue
            key = line.split(':')[0]
            values = line.split(':')[1].split(',')
            values = [val.strip() for val in values]
            descrip_tags[key] = values

    def fit(self, X, y):
        return self

    def transform(self, X):
        # import ipdb; ipdb.set_trace()
        X.fillna('no info', inplace=True)
        #v***vMy webscraper scraped these fields as lists, but I
        #v***vhad to separate them with ';' them when I wrote to csv
        X['Description'] = X['Description'].apply(lambda x: x.split(';'))
        X['Offense Code'] = X['Offense Code'].apply(lambda x: x.split(';'))
        #^***^
        X['Number of Offenses'] = X['Offense Code'].apply(lambda x: len(x))
        X['Priors'] = X['Description'].apply(lambda x: sum(1 if 'PRIOR' in y else 0 for y in x))
        X['Height in Inches'] = X['Height'].apply(lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1].strip('"')))
        X['BMI'] = X['Weight'] * 0.45 / (X['Height in Inches'] * 0.025) ** 2
        X['Female'] = pd.get_dummies(X['Sex'])['FEMALE']
        X['Transient'] = X['redtext0'].str.contains('TRANSIENT') * 1
        X['Age'] = 2016 - X['Date of Birth'].apply(lambda x: int(x.split('-')[-1]))
        X['Years in Violation'] = X['redtext1'].apply(lambda x: 2016 - int(x.split('/')[-1]
                                                        .split('.')[0]) if 'VIOLATION' in x else 0)
        X['SVP'] = X['redtext1'].str.contains('VIOLENT') * 1
        X['Age in Question'] = X['Age'] - X['Years in Violation']
        X['Possible Sentences'] = X['Offense Code'].apply(lambda lst: set.union(set(),
                                                    *[self.code_sentencing[code]
                                                    for code in lst if code in self.code_sentencing]))
        X['Minimum Sentence'] = X['Possible Sentences'].apply(lambda x: min(x) if len(x) > 0 else 0.0)
        X['Maximum Sentence'] = X['Possible Sentences'].apply(lambda x: max(x) if len(x) > 0 else 0.0)

        races = list(X['Ethnicity'].unique())
        races.remove('OTHER')
        X[races] = pd.get_dummies(X['Ethnicity'])[races]

        for key, value in self.descrip_tags.iteritems():
            X[key] = X['Description'].apply(lambda x: tag_counter(x, value))
        return X

class ColumnFilter(CustomMixin):
    exclude1 = [u'Description', u'Offense Code', u'Score', u'Score Date',
                u'Tool Name', u'Year of Last Conviction', u'Year of Last Release',
                u'redtext0', u'redtext1', u'redtext2', u'so_id', u'Date of Birth',
                u'Ethnicity', u'Eye Color', u'Hair Color', u'Height', 'Age', 'Violation',
                u'Last Known Address', u'Sex', 'Possible Sentences', 'Years in Violation']
    exclude2 = ['Age in Question', u'BLACK', u'HISPANIC', u'FILIPINO', u'OTHER ASIAN',
                u'CHINESE', u'PACIFIC ISLANDER', u'WHITE', u'UNKNOWN', u'GUAMANIAN',
                u'KOREAN', u'VIETNAMESE', u'AMERICAN INDIAN', u'ASIAN INDIAN', u'SAMOAN',
                u'HAWAIIAN', u'CAMBODIAN', u'JAPANESE', u'LAOTIAN', u'Height in Inches',
                u'BMI', u'Female', 'Weight']

    def __init__(self, prejudice=True):
        self.prejudice = prejudice

    def get_params(self, **kwargs):
        return {'prejudice':self.prejudice}

    def fit(self, X, y):
        return self

    def transform(self, X):
        X.drop(self.exclude1, axis=1, inplace=True)
        if not self.prejudice:
            X.drop(self.exclude2, axis=1, inplace=True)
        return X

# class Filter(CustomMixin):
#     exclude = ['Age in Question', u'BLACK', u'HISPANIC',
#        u'FILIPINO', u'OTHER ASIAN', u'CHINESE', u'PACIFIC ISLANDER', u'WHITE',
#        u'UNKNOWN', u'GUAMANIAN', u'KOREAN', u'VIETNAMESE', u'AMERICAN INDIAN',
#        u'ASIAN INDIAN', u'SAMOAN', u'HAWAIIAN', u'CAMBODIAN', u'JAPANESE',
#        u'LAOTIAN', u'Height in Inches', u'BMI', u'Female', 'Weight']
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X):
#         X.drop(self.exclude, axis=1, inplace=True)
#         return X


if __name__ == '__main__':
    df = pd.read_csv('data/Megans_List.csv')
    df['Violation'] = df['redtext1'].str.contains('VIOLATION') * 1
    y = df['Violation']
    y.fillna(0, inplace=True)

    p = Pipeline([
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter())
    ])

    transform = p.fit_transform(df.copy(), y)
    # model = p.fit(df, y)
    # transform = model.transform(df)

    p.set_params(columns__prejudice=False)

    transform_behavior = p.fit_transform(df.copy(), y)
    # model_behavior = p.fit(df, y)
    # transform_behavior = model_behavior.transform(df)

    # scale = StandardScaler()
    # transform_scaled = pd.DataFrame()
    # transform_scaled[transform.columns] = pd.DataFrame(scale.fit_transform(transform))
    #
    # lr = Logit(y, transform)
    # model_unscaled = lr.fit()
    # lr_scaled = Logit(y, transform_scaled)
    # model_scaled = lr_scaled.fit()
    #
    # rf = RandomForestClassifier()
    # rf_low_depth = RandomForestClassifier(max_depth=3)
    # rf.fit(transform, y)
    # rf_low_depth.fit(transform, y)
    #
    # def print_importance(rf, df):
    #     importances = zip(df.columns, rf.feature_importances_)
    #     for impor in sorted(importances, key=lambda x: x[1], reverse=True):
    #         print impor
    #
    # print '\nUnscaled: \n'
    # print model_unscaled.summary()
    # print '\nScaled: \n'
    # print model_scaled.summary()
    # print '\nImportances: \n'
    # print_importance(rf, transform)
    # print '\nLow Depth Importances: \n'
    # print_importance(rf_low_depth, transform)
