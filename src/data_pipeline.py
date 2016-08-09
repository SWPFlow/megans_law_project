from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, precision_score, fbeta_score, accuracy_score
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
    y = pd.Series(y)
    y.name = name

    return X, y

def get_train_test_sets(p, df, y):
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
    X_train_over, y_train_over = oversample(X_train, y_train)

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
        for scale_setting in ['noscale']: # removed 'scale' from list
            p.set_params(scale__scale=data_settings[scale_setting])
            test_sets[prejudice_setting] = (p.fit_transform(X_test.copy(), y_test), y_test)
            for balance_setting in ['bal']: #removed 'imba' from list
                features, target = data_settings[balance_setting]
                training_sets[prejudice_setting] = (p.fit_transform(features.copy(), target), target)

    return training_sets, test_sets


def big_grid_search(training_sets, test_sets):
        classifier_names = ['rfc', 'ada', 'gbc', 'log']
        classifiers = {'rfc': RandomForestClassifier,
                       'ada': AdaBoostClassifier,
                       'gbc': GradientBoostingClassifier,
                       'log': LogisticRegression}
        param_grids = {'rfc': {'n_estimators': [100, 250, 500],
                               'max_features': ['auto', 5, 10]},
                       'ada': {'n_estimators': [100, 250, 500],
                               'learning_rate': [1, 0.5, 0.1]},
                       'gbc': {'n_estimators': [100, 250, 500],
                               'learning_rate': [1, 0.1, 0.01]},
                       'log': {'penalty': ['l1', 'l2'],
                               'C': [1, 0.1, 0.01]}}

        best_score = {'prej': 0, 'noprej': 0}
        best_estimator = {}
        all_ests = {}
        for classifier in classifier_names:
            print classifiers[classifier].__name__
            for key, val in training_sets.iteritems():
                print 'Training set: ', key
                gs = GridSearchCV(classifiers[classifier](), param_grids[classifier], verbose=1, scoring=my_fbeta)
                gs.fit(*val)
                print "CV FBeta: ", gs.best_score_
                all_ests[(key, classifier)] = gs.best_estimator_
                preds = gs.best_estimator_.predict(test_sets[key][0])
                y_true = test_sets[key][1]
                print '\nRecall: ', recall_score(y_true, preds)
                print 'Precision: ', precision_score(y_true, preds)
                print 'Accuracy: ', accuracy_score(y_true, preds)
                if classifier != 'log':
                    feat_imp = zip(val[0].columns, gs.best_estimator_.feature_importances_)
                    print 'Feature Importances:\n', sorted(feat_imp, key=lambda x: x[1], reverse=True)[:5]

                if gs.best_score_ > best_score[key]:
                    print "We've got a winner!"
                    best_estimator[key] = gs.best_estimator_
                    best_score[key] = gs.best_score_


        return best_score, best_estimator, all_ests

def oversample_train_test(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
    X_train_over, y_train_over = oversample(X_train, y_train)
    return X_train_over, X_test, y_train_over, y_test

def my_fbeta(estimator, X, y):
    beta = 10
    preds = estimator.predict(X)
    return fbeta_score(y, preds, beta)



if __name__ == '__main__':
    df = pd.read_csv('data/Megans_List.csv')
    df['Violation'] = df['redtext1'].str.contains('VIOLATION') * 1
    y = df['Violation']
    y.fillna(0, inplace=True)

    feature_engineering = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('scale', ScaleOrNo())
    ])



    # Find out which training_sets and classifiers get the best recall.
    training_sets, test_sets = get_train_test_sets(feature_engineering, df, y)
    best_score, best_estimator, all_ests = big_grid_search(training_sets, test_sets)

    print '\nPrejudice: \n'
    prej_est = best_estimator['prej']
    prej_y = test_sets['prej'][1]
    prej_preds = prej_est.predict(test_sets['prej'][0])
    print 'Recall: ', recall_score(prej_y, prej_preds)
    print 'Precision: ', precision_score(prej_y, prej_preds)
    print 'Accuracy: ', accuracy_score(prej_y, prej_preds)

    print '\nNo Prejudice: \n'
    prej_est = best_estimator['noprej']
    prej_y = test_sets['noprej'][1]
    prej_preds = prej_est.predict(test_sets['noprej'][0])
    print 'Recall: ', recall_score(prej_y, prej_preds)
    print 'Precision: ', precision_score(prej_y, prej_preds)
    print 'Accuracy: ', accuracy_score(prej_y, prej_preds)

    with open('models.pkl', 'w') as f:
        pickle.dump(best_estimator, f)


    ## More specific grid_search
    # X_train, X_test, y_train, y_test = oversample_train_test(df, y)
    #
    # prej_model = Pipeline([
    #     ('lists', ListSplitter()),
    #     ('race', RaceDummies()),
    #     ('crime_sentence', CrimeAndSentence()),
    #     ('feat_eng', FeatureEngineer()),
    #     ('columns', ColumnFilter()),
    #     ('rf', RandomForestClassifier(n_jobs=-1))
    # ])
    #
    # no_prej_model = Pipeline([
    #     ('lists', ListSplitter()),
    #     ('race', RaceDummies()),
    #     ('crime_sentence', CrimeAndSentence()),
    #     ('feat_eng', FeatureEngineer()),
    #     ('columns', ColumnFilter(prejudice=False)),
    #     ('ada', AdaBoostClassifier())
    # ])
    #
    # print '\nPrejudice:\n'
    # prej_grid = {'rf__criterion': ['gini', 'entropy'],
    #              'rf__n_estimators': [200, 225, 250, 275, 300]}
    #
    # gs_prej = GridSearchCV(prej_model, prej_grid, scoring='recall')
    # gs_prej.fit(X_train.copy(), y_train)
    # print gs_prej.best_params_
    # prej_preds = gs_prej.best_estimator_.predict(X_test.copy())
    # print 'Recall: ', recall_score(y_test, prej_preds)
    # print 'Precision: ', precision_score(y_test, prej_preds)
    #
    # print '\nNo Prejudice:\n'
    #
    # no_prej_grid = {'ada__learning_rate': [0.05, 0.01, 0.005, 0.001],
    #                 'ada__n_estimators': [25, 50, 75, 100]}
    #
    # gs_no_prej = GridSearchCV(no_prej_model, no_prej_grid, scoring='recall')
    # gs_no_prej.fit(X_train.copy(), y_train)
    # print gs_no_prej.best_params_
    # no_prej_preds = gs_no_prej.best_estimator_.predict(X_test.copy())
    # print 'Recall: ', recall_score(y_test, no_prej_preds)
    # print 'Precision: ', precision_score(y_test, no_prej_preds)
