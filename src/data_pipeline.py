from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from statsmodels.discrete.discrete_model import Logit
from time import time
import pandas as pd
import numpy as np
import cPickle as pickle

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


class ListSplitter(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X.fillna('no info', inplace=True)
        #v***vMy webscraper scraped these fields as lists, but I
        #v***vhad to separate them with ';' them when I wrote to csv
        X['Description'] = X['Description'].apply(lambda x: x.split(';'))
        X['Offense Code'] = X['Offense Code'].apply(lambda x: x.split(';'))
        #^***^
        return X


class RaceDummies(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        races = list(X['Ethnicity'].unique())
        races.remove('OTHER')
        X[races] = pd.get_dummies(X['Ethnicity'])[races]
        return X


class CrimeAndSentence(CustomMixin):
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
        X['Possible Sentences'] = X['Offense Code'].apply(lambda lst: set.union(set(),
                                                    *[self.code_sentencing[code]
                                                    for code in lst if code in self.code_sentencing]))
        X['Minimum Sentence'] = X['Possible Sentences'].apply(lambda x: min(x) if len(x) > 0 else 0.0)
        X['Maximum Sentence'] = X['Possible Sentences'].apply(lambda x: max(x) if len(x) > 0 else 0.0)

        for key, value in self.descrip_tags.iteritems():
            X[key] = X['Description'].apply(lambda x: tag_counter(x, value))
        return X


class FeatureEngineer(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
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
        X['Constant'] = 1
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

class ScaleOrNo(CustomMixin):
    def __init__(self, scale=True):
        self.scale = scale
        self.scaler = StandardScaler()

    def fit(self, X, y):
        if self.scale:
            self.scaler.fit(X)
        return self

    def transform(self, X):
        if self.scale:
            self.scaler.transform(X)
        return X


def oversample(X, y):
    columns = X.columns
    name = y.name
    X = X.values
    y = y.values
    if y.sum() * 1.0 / y.size > .5:
        minority = 0
        sample_add = y.sum() * 2 - y.size
    else:
        minority = 1
        sample_add = y.size - y.sum() * 2

    print 'Minority Class: ', minority
    print 'Samples to Add: ', sample_add
    minority_index = np.arange(y.size)[y == minority]
    print 'Oversampling...'
    start = time()
    for i in xrange(sample_add):
        index = np.random.choice(minority_index)
        if i%1000 == 0 and i != 0:
            print i, 'added.\nElapsed Time: ', time() - start, 'seconds'
        X = np.append(X, X[index].reshape((1, len(columns))), axis=0)
        y = np.append(y, minority)
    X = pd.DataFrame(X)
    X.columns = columns
    y = pd.DataFrame(y)
    y.name = name

    return X, y





if __name__ == '__main__':
    df = pd.read_csv('data/Megans_List.csv')
    df['Violation'] = df['redtext1'].str.contains('VIOLATION') * 1
    y = df['Violation']
    y.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
    X_train_over, y_train_over = oversample(X_train, y_train)

    p = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('scale', ScaleOrNo())
    ])
    data_settings = {'prej': True,
                     'noprej': False,
                     'imba': (X_train, y_train),
                     'bal': (X_train_over, y_train_over),
                     'scale': True,
                     'noscale': False}
    training_sets = {}
    test_sets = {}
    for prejudice_setting in ['prej', 'noprej']:
        p.set_params(columns__prejudice=data_settings[prejudice_setting])
        for balance_setting in ['imba', 'bal']:
            features, target = data_settings[balance_setting]
            for scale_setting in ['scale', 'noscale']:
                p.set_params(scale__scale=data_settings[scale_setting])
                training_sets[(prejudice_setting,
                           balance_setting,
                           scale_setting)] = (p.fit_transform(features.copy(), target), target)
                test_sets[(prejudice_setting,
                           scale_setting)] = (p.fit_transform(X_test.copy(), y_test), y_test)

    classifier_names = ['rfc', 'ada', 'gbc', 'svc', 'log', 'gnb']
    classifiers = {'rfc': RandomForestClassifier,
                   'ada': AdaBoostClassifier,
                   'gbc': GradientBoostingClassifier,
                   'svc': SVC,
                   'log': LogisticRegression,
                   'gnb': GaussianNB}
    param_grids = {'rfc': {'n_estimators': [100, 250, 500],
                           'max_features': ['auto', 5, 10],
                           'max_depth': [None, 3, 5, 7]},
                   'ada': {'n_estimators': [50, 100, 250],
                           'learning_rate': [1, 0.1, 0.01]},
                   'gbc': {'n_estimators': [100, 250, 500],
                           'learning_rate': [1, 0.1, 0.01]},
                   'svc': [{'kernel': ['linear'],
                            'C': [10, 1, 0.1, 0.01]},
                           {'kernel': ['rbf'],
                           'C': [10, 1, 0.1, 0.01],
                           'gamma': ['auto', 1.0, 0.1, 0.001]},
                           {'kernel': ['poly'],
                           'C': [10, 1, 0.1, 0.01],
                           'degree': [3, 5, 7]}],
                   'log': {'penalty': ['l1', 'l2'],
                           'C': [1, 0.1, 0.01]},
                   'gnb': {'class_prior_': [[0.8, 0.2], [0.9, 0.1], [0.5, 0.5]]}}

    best_score = {'prej': 0, 'noprej': 0}
    best_estimator, best_train_set, best_classifier = {}, {}, {}
    for classifier in classifier_names:
        print classifiers[classifier].__name__
        for key, val in training_sets.iteritems():
            print 'Training set: ', key
            gs = GridSearchCV(classifiers[classifier](), param_grids[classifier], scoring='recall')
            gs.fit(val[0], val[1])
            if gs.best_score_ > best_score[key[0]]:
                best_estimator[key[0]] = gs.best_estimator_
                best_score[key[0]] = gs.best_score_
                best_train_set[key[0]] = key
                best_classifier[key[0]] = classifier

    with open('results.pkl', 'w') as f:
        pickle.dump((best_score, best_estimator, best_train_set, best_classifier), f)

    # gscv = GridSearchCV(p, param_grid, scoring='recall')

    # transform = p.fit_transform(df.copy(), y)
    # model = p.fit(df, y)
    # transform = model.transform(df)

    # p.set_params(columns__prejudice=False)
    #
    # transform_behavior = p.fit_transform(df.copy(), y)
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
