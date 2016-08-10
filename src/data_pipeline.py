from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, precision_score, fbeta_score, accuracy_score, precision_recall_curve, auc
from statsmodels.discrete.discrete_model import Logit
from time import time
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

beta = 1

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

class PickEstimator(BaseEstimator):
    def __init__(self, estimator=AdaBoostClassifier()):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X,y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

class EnsembleEstimator(BaseEstimator):
    def fit(self, X, y):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.rf.fit(X, y)
        self.ada = AdaBoostClassifier(n_estimators=500)
        self.ada.fit(X, y)
        self.gbc = GradientBoostingClassifier(n_estimators=500)
        self.gbc.fit(X, y)
        self.log = LogisticRegression()
        self.log.fit(X, y)
        return self

    def predict(self, X):
        return self.predict_proba(X)[:,1] > .5

    def predict_proba(self, X):
        rf_proba = self.rf.predict_proba(X)
        ada_proba = self.ada.predict_proba(X)
        gbc_proba = self.gbc.predict_proba(X)
        log_proba = self.log.predict_proba(X)
        scale = MinMaxScaler()
        rf_proba = scale.fit_transform(rf_proba)
        ada_proba = scale.fit_transform(ada_proba)
        gbc_proba = scale.fit_transform(gbc_proba)
        log_proba = scale.fit_transform(log_proba)
        proba = (rf_proba + ada_proba + gbc_proba + log_proba) / 4
        return proba

def oversample(X, y):
    if y.sum() * 1.0 / y.size > .5:
        minority = 0
        sample_add = y.sum() * 2 - y.size
    else:
        minority = 1
        sample_add = y.size - y.sum() * 2

    X_overs = X[y==1].sample(sample_add, replace=True)
    y_overs = y[y==1].sample(sample_add, replace=True)
    X = pd.concat([X, X_overs])
    y = pd.concat([y, y_overs])
    # minority_index = np.arange(y.size)[y == minority]
    # print 'Oversampling...'
    # start = time()
    # for i in xrange(sample_add):
    #     index = np.random.choice(minority_index)
    #     if i%1000 == 0 and i != 0:
    #         print i, 'added.\nElapsed Time: ', time() - start, 'seconds'
    #     X = np.append(X, X[index].reshape((1, len(columns))), axis=0)
    #     y = np.append(y, minority)
    # X = pd.DataFrame(X)
    # X.columns = columns
    # y = pd.Series(y)
    # y.name = name

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
            print '\n', classifiers[classifier].__name__
            for key, val in training_sets.iteritems():
                print '\nTraining set: ', key
                gs = GridSearchCV(classifiers[classifier](), param_grids[classifier], verbose=2, scoring=pr_auc)
                gs.fit(*val)
                print "\nCV PR_AUC: ", gs.best_score_
                all_ests[(key, classifier)] = gs.best_estimator_
                preds = gs.best_estimator_.predict(test_sets[key][0])
                y_true = test_sets[key][1]
                print 'Recall: ', recall_score(y_true, preds)
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
    beta = .5
    preds = estimator.predict(X)
    return fbeta_score(y, preds, beta)

def change_beta(new_beta):
    global beta
    beta = new_beta

def more_precise_grid(df, y):
        X_train, X_test, y_train, y_test = oversample_train_test(df, y)

        prej_model = Pipeline([
            ('lists', ListSplitter()),
            ('race', RaceDummies()),
            ('crime_sentence', CrimeAndSentence()),
            ('feat_eng', FeatureEngineer()),
            ('columns', ColumnFilter()),
            ('est', PickEstimator())
        ])

        no_prej_model = Pipeline([
            ('lists', ListSplitter()),
            ('race', RaceDummies()),
            ('crime_sentence', CrimeAndSentence()),
            ('feat_eng', FeatureEngineer()),
            ('columns', ColumnFilter(prejudice=False)),
            ('est', PickEstimator())
        ])

        for esto in [LogisticRegression(),
                     AdaBoostClassifier(n_estimators=500),
                     GradientBoostingClassifier(n_estimators=500),
                     RandomForestClassifier(n_estimators=100)]:
            prej_model.set_params(est__estimator=esto)
            no_prej_model.set_params(est__estimator=esto)

            print '\nPrejudice:\n'
            # prej_grid = {'gbc__learning_rate': [1.25, 1, 0.75],
            #              'gbc__n_estimators': [750]}
            #
            # gs_prej = GridSearchCV(prej_model, prej_grid, scoring=my_fbeta, verbose=1)
            # gs_prej.fit(X_train.copy(), y_train)
            # print '\nCV FBeta: ', gs_prej.best_score_
            # print gs_prej.best_params_
            prej_model.fit(X_train.copy(), y_train)
            prej_preds = prej_model.predict(X_test.copy())
            prej_scores = prej_model.predict_proba(X_test.copy())[:,1]
            print 'Proba Range: {}-{}'.format(prej_scores.min(), prej_scores.max())
            print 'Recall: ', recall_score(y_test, prej_preds)
            print 'Precision: ', precision_score(y_test, prej_preds)
            print 'Accuracy: ', accuracy_score(y_test, prej_preds)
            plot_pr_curve(prej_scores, y_test, 'Prej, {}'.format(esto.__class__.__name__))

            print '\nNo Prejudice:\n'

            no_prej_grid = {'gbc__learning_rate': [0.07, 0.05, 0.04],
                            'gbc__n_estimators': [75, 80, 85]}

            # gs_no_prej = GridSearchCV(no_prej_model, no_prej_grid, scoring=my_fbeta, verbose=1)
            # gs_no_prej.fit(X_train.copy(), y_train)
            # print '\nCV FBeta: ', gs_no_prej.best_score_
            # print gs_no_prej.best_params_
            no_prej_model.fit(X_train.copy(), y_train)
            no_prej_preds = no_prej_model.predict(X_test.copy())
            no_prej_scores = no_prej_model.predict_proba(X_test.copy())[:,1]
            print 'Proba Range: {}-{}'.format(no_prej_scores.min(), no_prej_scores.max())
            print 'Recall: ', recall_score(y_test, no_prej_preds)
            print 'Precision: ', precision_score(y_test, no_prej_preds)
            print 'Accuracy: ', accuracy_score(y_test, no_prej_preds)

            plot_pr_curve(no_prej_scores, y_test, 'No Prej, {}'.format(esto.__class__.__name__))

        plt.legend(loc='best')
        plt.show()

def plot_pr_curve(probas, y_test, label):
    precision, recall, thresholds = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, label=label)

def pr_auc(estimator, X, y):
    probas = estimator.predict_proba(X.copy())[:,1]
    precision, recall, thresholds = precision_recall_curve(y, probas)
    return auc(recall, precision)


def proba_df_maker(df, y):
    X_train, X_test, y_train, y_test = oversample_train_test(df, y)

    prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('est', PickEstimator())
    ])

    no_prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter(prejudice=False)),
        ('est', PickEstimator())
    ])

    feature_engineering = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter()),
        ('scale', ScaleOrNo())
    ])

    new_df = feature_engineering(X_test.copy())
    new_df['Violation'] = y_test.copy()
    for esto in [LogisticRegression(),
                 AdaBoostClassifier(n_estimators=500),
                 GradientBoostingClassifier(n_estimators=500),
                 RandomForestClassifier(n_estimators=100)]:
        prej_model.set_params(est__estimator=esto)
        no_prej_model.set_params(est__estimator=esto)

        print '\nPrejudice:\n'
        prej_model.fit(X_train.copy(), y_train)
        prej_scores = prej_model.predict_proba(X_test.copy())[:,1]
        new_df['Prej {}'.format(esto.__class__.__name__)] = prej_scores

        print '\nNo Prejudice:\n'
        no_prej_model.fit(X_train.copy(), y_train)
        no_prej_scores = no_prej_model.predict_proba(X_test.copy())[:,1]
        new_df['No Prej {}'.format(esto.__class__.__name__)] = no_prej_scores

    return new_df


def test_ensemble(df, y):
    X_train, X_test, y_train, y_test = oversample_train_test(df, y)

    no_prej_model = Pipeline([
        ('lists', ListSplitter()),
        ('race', RaceDummies()),
        ('crime_sentence', CrimeAndSentence()),
        ('feat_eng', FeatureEngineer()),
        ('columns', ColumnFilter(prejudice=False)),
        ('est', EnsembleEstimator())
    ])

    for esto in [EnsembleEstimator(),
                 GradientBoostingClassifier(n_estimators=500)]:
        no_prej_model.fit(X_train.copy(), y_train)
        no_prej_preds = no_prej_model.predict(X_test.copy())
        no_prej_scores = no_prej_model.predict_proba(X_test.copy())[:,1]
        print 'Proba Range: {}-{}'.format(no_prej_scores.min(), no_prej_scores.max())
        print 'Recall: ', recall_score(y_test, no_prej_preds)
        print 'Precision: ', precision_score(y_test, no_prej_preds)
        print 'Accuracy: ', accuracy_score(y_test, no_prej_preds)

        plot_pr_curve(no_prej_scores, y_test, 'No Prej {}'.format(esto.__class__.__name__))
    plt.legend(loc='best')
    plt.show()





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


    # # Find out which training_sets and classifiers get the best fbeta.
    # training_sets, test_sets = get_train_test_sets(feature_engineering, df, y)
    # best_score, best_estimator, all_ests = big_grid_search(training_sets, test_sets)


    # # More specific grid_search
    # more_precise_grid(df, y)

    # # Make a df with the probabilities
    # proba_df = proba_df_maker(df, y)

    # Test ensemble model
    test_ensemble(df, y)
